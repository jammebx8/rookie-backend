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

Your job is NOT to sound like a textbook or coaching module.
Your job is to sound like a real person explaining naturally.

The user is already talking to {req.buddy_name}, so fully stay in that character's personality, tone, slang, emotional style, teasing level, and teaching style.

IMPORTANT:
- Do not sound robotic
- Do not sound overly polished
- Do not use textbook phrases like "Firstly", "Hence", "Therefore", "We know that"
- Do not overexplain
- Do not repeat the question
- Do not copy the provided solution word for word
- Sound like you are thinking through the question naturally with the student
- Every step should feel conversational
- Use short natural transitions like:
  - "Dekho"
  - "So"
  - "Ab"
  - "Yaha pe"
  - "Simple hai"
  - "Notice karo"
  - "Bas yahi trick hai"
  - "Key idea yeh hai"
- If the question is easy, acknowledge that casually
- If the question is tricky, mention where students usually get confused
- Occasionally add small emotional reactions that match the character
- Keep the tone dynamic, not flat
{pct}

Question:
{req.question_text}

Options:
A) {req.option_A}
B) {req.option_B}
C) {req.option_C}
D) {req.option_D}

Correct Answer: {req.correct_option}

Reference Solution:
{req.solution}

OUTPUT FORMAT:
- Maximum 6 steps
- Each step should be 1–2 short lines only
- Use plain conversational language
- Use LaTeX for all math
- Inline math with $...$
- Block equations with $$...$$
- Keep spacing clean
- End with one final line:
  Answer: Option {req.correct_option}

Now generate the explanation in {req.buddy_name}'s exact personality."""
        else:
            prompt = f"""You are a friendly JEE tutor explaining a solution to a student. Your goal is to make it feel personal, easy, and short — like a smart friend talking them through it, not a textbook.

Use the solution logic below as your reference. Rephrase it naturally — do NOT copy it word for word.
{pct}

Question: {req.question_text}

Options:
A) {req.option_A}
B) {req.option_B}
C) {req.option_C}
D) {req.option_D}

Correct Answer: {req.correct_option}

Solution Logic (reference only — rewrite naturally):
{req.solution}

RULES:
- Max 6 short steps (1–2 lines each)
- Use natural transitions: "So", "Notice that", "This gives us", "The key idea is"
- Sound like a person, not a formal document
- NO unnecessary repetition or padding
- Use $ for inline math, $$ for block equations (each on its own line, blank line before and after)
- End with: Answer: Option {req.correct_option}

Write the solution now:"""

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
