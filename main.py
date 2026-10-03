"""
Rookie FastAPI Backend
======================
Single /solution endpoint replacing the Next.js route.
All four actions are supported:
  - generate_solution
  - explain_5yr
  - better_understanding
  - dig_deeper


The `determine_answer` action has been removed — `correct_option` now
arrives directly from the database.
"""

from __future__ import annotations
import os
import json
import re
from typing import Any, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY: str = os.environ["GROQ_API_KEY"]
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-oss-120b"

app = FastAPI(title="Rookie Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


# ── Request / Response schemas ─────────────────────────────────────────────

class SolutionRequest(BaseModel):
    action: str

    # ── Question data (from DB row) ────────────────────────────────────────
    question_id:        Optional[str] = None
    question_text:      Optional[str] = None
    option_A:           Optional[str] = None
    option_B:           Optional[str] = None
    option_C:           Optional[str] = None
    option_D:           Optional[str] = None
    option_a_img:       Optional[str] = None
    option_b_img:       Optional[str] = None
    option_c_img:       Optional[str] = None
    option_d_img:       Optional[str] = None
    correct_option:     Optional[str] = None   # comes from DB now
    solution:           Optional[str] = None
    question_img_url:   Optional[str] = None
    solution_image_url: Optional[str] = None
    exam_shift:         Optional[str] = None
    source_url:         Optional[str] = None
    subject:            Optional[str] = None
    chapter:            Optional[str] = None
    option_a_percent:   Optional[float] = None
    option_b_percent:   Optional[float] = None
    option_c_percent:   Optional[float] = None
    option_d_percent:   Optional[float] = None

    # ── AI Buddy fields ────────────────────────────────────────────────────
    buddy_id:           Optional[str] = None
    buddy_name:         Optional[str] = None
    buddy_system_prompt: Optional[str] = None

    # ── Model config ───────────────────────────────────────────────────────
    model:       str   = Field(default=DEFAULT_MODEL)
    temperature: float = Field(default=0.3)
    max_tokens:  int   = Field(default=500)


# ── Groq helper ────────────────────────────────────────────────────────────

async def call_groq(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    """Non-blocking Groq call via httpx async client."""
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            GROQ_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {GROQ_API_KEY}",
            },
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
    if resp.status_code != 200:
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"Groq API error: {resp.text}",
        )
    return resp.json()


def build_messages(
    user_prompt: str,
    buddy_system_prompt: Optional[str],
) -> list[dict[str, str]]:
    """Prepend buddy system prompt when available."""
    if buddy_system_prompt:
        return [
            {"role": "system", "content": buddy_system_prompt},
            {"role": "user",   "content": user_prompt},
        ]
    return [{"role": "user", "content": user_prompt}]


def first_choice(groq_data: dict) -> str:
    return groq_data.get("choices", [{}])[0].get("message", {}).get("content", "")


# ── Option percentage context helper ──────────────────────────────────────

def pct_context(req: SolutionRequest) -> str:
    """
    If option-percentage data is available, return a one-liner that the
    AI can weave into its explanation (e.g. common wrong-answer analysis).
    """
    if any(v is not None for v in [
        req.option_a_percent, req.option_b_percent,
        req.option_c_percent, req.option_d_percent,
    ]):
        return (
            f"\nStudent response distribution — "
            f"A: {req.option_a_percent}%, "
            f"B: {req.option_b_percent}%, "
            f"C: {req.option_c_percent}%, "
            f"D: {req.option_d_percent}%"
        )
    return ""


# ══════════════════════════════════════════════════════════════════════════
# /solution  endpoint
# ══════════════════════════════════════════════════════════════════════════

@app.post("/solution")
async def solution(req: SolutionRequest):

    # ── GENERATE SOLUTION ─────────────────────────────────────────────────
    if req.action == "generate_solution":
        if not all([req.question_text, req.option_A, req.option_B,
                    req.option_C, req.option_D, req.solution]):
            raise HTTPException(
                status_code=400,
                detail="question_text, option_A/B/C/D, and solution are required",
            )

        pct = pct_context(req)

        if req.buddy_name and req.buddy_system_prompt:
          prompt = f"""A student needs help understanding a JEE question.

You are {req.buddy_name}, and you are personally teaching this student.

Your job is NOT to write a textbook solution, coaching-module solution, or answer-key explanation.

Your job is to make the student genuinely understand the question while sounding like a real human tutor with a distinct personality.

PERSONALITY
- Fully stay in {req.buddy_name}'s personality, tone, humour, energy, slang, emotional style, teasing level, and teaching style.
- The personality should be noticeable from the explanation itself.
- Do NOT simply add words like "Dekho", "So", or "Yaha pe" to make the response sound casual.
- Talk as if {req.buddy_name} is actually sitting with the student and explaining the question.
- Follow the buddy system prompt carefully.
- Do not sound like an AI tutor, textbook, coaching module, Wikipedia article, or answer key.
- Do not use the same catchphrases in every answer.
- Do not force jokes, slang, or emotional reactions when they don't fit.
- If the question is easy, react naturally and keep the explanation short.
- If the question is tricky, point out the exact trap.
- If the student's likely misconception is obvious, address it naturally.

EXPLANATION APPROACH
- First understand the actual mathematical/scientific reasoning yourself.
- Identify the core concept or trick behind the question.
- Explain the reasoning naturally from that core idea.
- Do NOT mechanically explain A, B, C, D one after another.
- Only discuss wrong options when they reveal a useful misconception or important trap.
- Do not repeat the question unnecessarily.
- Do not restate obvious information just to make the answer longer.
- Prefer intuition and reasoning over formal textbook definitions.
- If there is a shortcut, show it naturally.
- If there is no shortcut or trick, do not invent one.
- Keep the explanation concise but complete enough that the student understands WHY the answer is correct.
- Do not overexplain easy questions.
- Do not under-explain questions that require reasoning.
- Do not copy the reference solution's structure.
- Do not copy the reference solution's wording.
- Do not follow the reference solution's sentence structure.
- The reference solution is ONLY a source of information and reasoning.
- Rewrite everything naturally in your own words.

REFERENCE SOLUTION HANDLING
The reference solution may be:
- overly verbose
- repetitive
- poorly written
- textbook-like
- grammatically incorrect
- formatted badly
- mathematically formatted incorrectly
- missing useful intuition

Do not blindly follow it.

Use it to understand the intended solution and verify the reasoning, then create a better explanation for the student.

If the reference solution contains an obvious factual or logical inconsistency with the provided correct answer, carefully reason from the question and correct answer instead of blindly copying the inconsistent explanation.

LATEX ACCURACY
- Use clean, valid LaTeX for every mathematical expression.
- The reference solution may contain broken, malformed, incomplete, or incorrectly formatted LaTeX.
- NEVER copy broken LaTeX into your output.
- If the reference contains broken LaTeX, infer the intended mathematical meaning from context and write the correct LaTeX yourself.
- Fix missing braces.
- Fix incorrect LaTeX commands.
- Fix broken fractions.
- Fix incorrect superscripts and subscripts.
- Fix Greek letters and mathematical symbols.
- Fix units and mathematical notation when necessary.
- Do not preserve malformed backticks or broken mathematical markup.
- Use $...$ for inline mathematics.
- Use $$...$$ for display equations.
- Put display equations on their own lines.
- Do not mention that the reference solution contained broken LaTeX.
- The final response must contain clean mathematical notation even when the source does not.

NATURAL LANGUAGE
- Sound conversational and human.
- Use natural transitions when they genuinely fit:
  "Dekho"
  "So"
  "Ab"
  "Yaha pe"
  "Notice karo"
  "Simple hai"
  "Bas yahi catch hai"
  "Key idea ye hai"
  "Don't overthink this"
  "Ye classic trap hai"
- Do NOT force these phrases into every response.
- Avoid repetitive sentence patterns.
- Avoid:
  "Firstly"
  "Secondly"
  "Thirdly"
  "Hence"
  "Therefore"
  "We know that"
  "It can be observed that"
  "Thus, we can conclude"
  unless they are genuinely natural and necessary.
- Occasionally add a small emotional reaction when it fits the buddy's personality.
- Do not add fake enthusiasm to every question.
- Do not use generic filler such as "Let's dive into this", "Great question", or "Let's understand this step by step" unless it genuinely fits the character.

TEACHING QUALITY
- Focus on the concept the student should remember after solving the question.
- When useful, mention the common JEE trap or misconception.
- Make the explanation memorable rather than merely correct.
- If a definition is important, explain it naturally instead of just reciting it.
- If a formula is used, explain what it means when useful.
- Don't introduce unrelated theory.
- Don't solve a different problem from the one asked.

OPTION ANALYSIS
- The correct answer is provided by the database.
- Do not attempt to change the correct answer.
- Use the options to understand what misconception the question is testing.
- Do not waste space explaining obviously irrelevant wrong options.
- If several statements/options are involved, group related reasoning instead of producing repetitive A/B/C/D explanations.

{pct}

Question:
{req.question_text}

Options:
A) {req.option_A}
B) {req.option_B}
C) {req.option_C}
D) {req.option_D}

Correct Answer:
{req.correct_option}

Reference Solution:
{req.solution}

OUTPUT FORMAT
- Maximum 5 short paragraphs or steps.
- Each paragraph should be concise.
- Do not force a step-by-step structure if the question can be explained naturally in a few sentences.
- Use short paragraphs rather than huge blocks of text.
- Keep the answer proportional to the difficulty of the question.
- Easy question → short explanation.
- Difficult question → enough reasoning to understand it.
- Do not repeat the question.
- Do not include unnecessary headings such as "Solution", "Explanation", or "Analysis".
- End with exactly:
Answer: Option {req.correct_option}

Now explain the question naturally in {req.buddy_name}'s exact personality.
"""

        messages = build_messages(prompt, req.buddy_system_prompt)
        groq_data = await call_groq(messages, req.model, temperature=0.5, max_tokens=2000)
        return {"solution": first_choice(groq_data) or req.solution, "full_response": groq_data}

    # ── EXPLAIN LIKE A CHILD ──────────────────────────────────────────────
    elif req.action == "explain_5yr":
        if not all([req.question_text, req.solution]):
            raise HTTPException(
                status_code=400,
                detail="question_text and solution are required",
            )

        if req.buddy_name and req.buddy_system_prompt:
            user_prompt = f"""Explain the following JEE question solution in very simple terms, as if talking to a 10-year-old. Use analogies, simple language, and avoid technical jargon. Stay in your character voice.

Question: {req.question_text}
Correct Answer: {req.correct_option}
Solution: {req.solution}

Explain simply:"""
        else:
            user_prompt = f"""Explain the following JEE question solution in very simple terms that even a 5-year-old could understand. Use analogies, simple language, and avoid technical jargon.

Question: {req.question_text}
Correct Answer: {req.correct_option}
Solution: {req.solution}

Explain this solution in simple, friendly language:"""

        messages = build_messages(user_prompt, req.buddy_system_prompt)
        groq_data = await call_groq(messages, req.model, temperature=0.7, max_tokens=800)
        return {"explanation": first_choice(groq_data) or "Could not generate explanation.", "full_response": groq_data}

    # ── BETTER UNDERSTANDING ──────────────────────────────────────────────
    elif req.action == "better_understanding":
        if not all([req.question_text, req.solution]):
            raise HTTPException(
                status_code=400,
                detail="question_text and solution are required",
            )

        if req.buddy_name and req.buddy_system_prompt:
            user_prompt = f"""Give a simpler, more intuitive explanation of this JEE question solution. Focus on the core concept. Stay in your character voice and keep it concise.

Question: {req.question_text}
Correct Answer: {req.correct_option}
Solution: {req.solution}

Simpler explanation:"""
        else:
            user_prompt = f"""Provide a simpler, more intuitive explanation of this JEE question solution. Focus on the core concept and make it easier to understand.

Question: {req.question_text}
Correct Answer: {req.correct_option}
Solution: {req.solution}

Provide a clearer, more intuitive explanation:"""

        messages = build_messages(user_prompt, req.buddy_system_prompt)
        groq_data = await call_groq(messages, req.model, temperature=0.7, max_tokens=800)
        return {"explanation": first_choice(groq_data) or "Could not generate explanation.", "full_response": groq_data}

    # ── DIG DEEPER ────────────────────────────────────────────────────────
    elif req.action == "dig_deeper":
        if not all([req.question_text, req.solution]):
            raise HTTPException(
                status_code=400,
                detail="question_text and solution are required",
            )

        user_prompt = f"""Based on this JEE question and solution, create a simpler conceptual MCQ to test understanding of the underlying concept. Make it easier than the original question but focus on the same core concept.

Question: {req.question_text}
Correct Answer: {req.correct_option}
Solution: {req.solution}

Create a new MCQ and respond in this EXACT JSON format (no additional text, no markdown):
{{
  "question": "your question here",
  "options": ["option 1", "option 2", "option 3", "option 4"],
  "correctAnswer": "A",
  "explanation": "brief explanation of why this is the correct answer"
}}"""

        # No buddy system prompt for dig_deeper — we need clean JSON output
        groq_data = await call_groq(
            [{"role": "user", "content": user_prompt}],
            req.model,
            temperature=0.7,
            max_tokens=1000,
        )
        ai_response = first_choice(groq_data)

        json_match = re.search(r"\{[\s\S]*\}", ai_response)
        if not json_match:
            raise HTTPException(
                status_code=500,
                detail={"error": "Could not parse MCQ JSON from AI response", "ai_response": ai_response},
            )

        mcq_data = json.loads(json_match.group())
        return {"mcq": mcq_data, "raw_response": ai_response}

    # ── UNKNOWN ACTION ────────────────────────────────────────────────────
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown action '{req.action}'. Valid actions: generate_solution, explain_5yr, better_understanding, dig_deeper",
        )
