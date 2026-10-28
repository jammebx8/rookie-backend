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
import base64
from typing import Any, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY")
DEEPGRAM_API_KEY: str = os.environ.get("DEEPGRAM_API_KEY", "745155e2a953bd317278905804c8d3f50b2eee83")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_TTS_URL = "https://api.groq.com/openai/v1/audio/speech"
DEEPGRAM_URL = "https://api.deepgram.com/v1/listen?model=nova-3&language=en&smart_format=true"

VISION_MODEL = "qwen/qwen3.8-27b"
REASONING_MODEL = "openai/gpt-oss-120b"
DEFAULT_MODEL = "openai/gpt-oss-120b"
TTS_MODEL = "canopylabs/orpheus-v1-english"

app = FastAPI(title="Rookie Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Payment routes ─────────────────────────────────────────────────────────────
from payment import router as payment_router   # noqa: E402
app.include_router(payment_router)


@app.get("/")
async def health():
    return {"status": "ok", "service": "Rookie Backend"}


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
- ONLY use these two delimiter styles — nothing else:
  - Inline math:   $...$       (single dollar signs, no space after opening $)
  - Display math:  $$...$$     (double dollar signs, on their own line)
- NEVER use \[...\] or \(...\) — these are not supported by the renderer.
- NEVER use \begin{{equation}}, \begin{{align}}, or any LaTeX environment blocks.
- Every mathematical symbol, variable, fraction, subscript, superscript, Greek letter, or formula MUST be inside $ or $$ delimiters.
- Do not write bare math like mgsinθ or a_s outside of $ delimiters.
- Put display equations ($$...$$) on their own line with a blank line before and after.
- Do not mention that the reference solution contained broken LaTeX.
- The final response must contain clean, correctly delimited mathematical notation.

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

MATH FORMATTING: Use only $...$ for inline math and $$...$$ for display math. Never use \\[...\\] or \\(...\\).

Question: {req.question_text}
Correct Answer: {req.correct_option}
Solution: {req.solution}

Explain simply:"""
        else:
            user_prompt = f"""Explain the following JEE question solution in very simple terms that even a 5-year-old could understand. Use analogies, simple language, and avoid technical jargon.

MATH FORMATTING: Use only $...$ for inline math and $$...$$ for display math. Never use \\[...\\] or \\(...\\).

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

MATH FORMATTING: Use only $...$ for inline math and $$...$$ for display math. Never use \\[...\\] or \\(...\\).

Question: {req.question_text}
Correct Answer: {req.correct_option}
Solution: {req.solution}

Simpler explanation:"""
        else:
            user_prompt = f"""Provide a simpler, more intuitive explanation of this JEE question solution. Focus on the core concept and make it easier to understand.

MATH FORMATTING: Use only $...$ for inline math and $$...$$ for display math. Never use \\[...\\] or \\(...\\).

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


# ══════════════════════════════════════════════════════════════════════════
# SNAP & SOLVE + VOICE AI TUTOR ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════

def clean_for_speech(text: str) -> str:
    """Converts LaTeX and markdown math into natural spoken words for TTS."""
    s = text
    # Replace common LaTeX math commands with natural spoken phrases
    s = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'\1 over \2', s)
    s = re.sub(r'\\sqrt\{([^}]+)\}', r'square root of \1', s)
    s = re.sub(r'\\sqrt\[(\d+)\]\{([^}]+)\}', r'\1th root of \2', s)
    s = re.sub(r'\\boxed\{([^}]+)\}', r'\1', s)
    s = re.sub(r'\\text\{([^}]+)\}', r'\1', s)
    s = re.sub(r'\\mathrm\{([^}]+)\}', r'\1', s)
    s = re.sub(r'\\mathbf\{([^}]+)\}', r'\1', s)
    
    # Symbols & operators
    s = s.replace(r'\times', ' times ')
    s = s.replace(r'\pm', ' plus or minus ')
    s = s.replace(r'\approx', ' approximately ')
    s = s.replace(r'\neq', ' not equal to ')
    s = s.replace(r'\leq', ' less than or equal to ')
    s = s.replace(r'\le', ' less than or equal to ')
    s = s.replace(r'\geq', ' greater than or equal to ')
    s = s.replace(r'\ge', ' greater than or equal to ')
    s = s.replace(r'\rightarrow', ' approaches ')
    s = s.replace(r'\Rightarrow', ' implies ')
    s = s.replace(r'\infty', ' infinity ')
    s = s.replace(r'\Delta', ' delta ')
    s = s.replace(r'\theta', ' theta ')
    s = s.replace(r'\pi', ' pi ')
    s = s.replace(r'\alpha', ' alpha ')
    s = s.replace(r'\beta', ' beta ')
    s = s.replace(r'\gamma', ' gamma ')
    s = s.replace(r'\lambda', ' lambda ')
    s = s.replace(r'\mu', ' micro ')
    s = s.replace(r'\sigma', ' sigma ')
    s = s.replace(r'\omega', ' omega ')
    
    # Powers and subscripts
    s = re.sub(r'\^2\b', ' squared', s)
    s = re.sub(r'\^3\b', ' cubed', s)
    s = re.sub(r'\^\{?([0-9a-zA-Z]+)\}?', r' to the power of \1', s)
    s = re.sub(r'_\{?([0-9a-zA-Z]+)\}?', r' sub \1', s)
    
    # Strip delimiters and markdown
    s = re.sub(r'\$\$|\$', ' ', s)
    s = re.sub(r'\\\[|\\\]|\\\(|\\\)', ' ', s)
    s = re.sub(r'[\*_#`~]', ' ', s)
    s = re.sub(r'\\[a-zA-Z]+', ' ', s)
    s = re.sub(r'[{}]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


class SnapAnalyzeRequest(BaseModel):
    image_base64: str


class SnapTranscribeRequest(BaseModel):
    audio_base64: str
    mime_type: Optional[str] = "audio/webm"


class SnapVoiceDoubtRequest(BaseModel):
    question_text: str
    solution: str
    user_doubt: str
    history: Optional[list[dict[str, str]]] = Field(default_factory=list)
    voice: Optional[str] = "autumn"


class SnapTTSRequest(BaseModel):
    text: str
    voice: Optional[str] = "autumn"


@app.post("/snap-solve/analyze")
async def snap_analyze(req: SnapAnalyzeRequest):
    """
    1. Inspects image with qwen/qwen3.8-27b on Groq.
    2. Verifies whether the image contains schoolwork/academic doubt.
       If not, returns is_schoolwork: False with error.
    3. If valid, extracts problem text and calls openai/gpt-oss-120b for
       a complete step-by-step solution rendered with LaTeX math.
    """
    if not req.image_base64:
        raise HTTPException(status_code=400, detail="image_base64 is required")

    # Format data URI
    img_b64 = req.image_base64.strip()
    if not img_b64.startswith("data:"):
        img_b64 = f"data:image/jpeg;base64,{img_b64}"

    # Step 1: Vision analysis with qwen/qwen3.8-27b
    vision_prompt = (
        "You are an expert AI academic scanner and vision analyzer for students.\n"
        "Carefully examine this cropped image.\n\n"
        "TASK 1: VALIDATION\n"
        "Determine if this image contains academic schoolwork, homework, a textbook/notebook page, "
        "a physics/chemistry/math/biology problem, an equation, or an exam question.\n"
        "If it is NOT schoolwork (e.g. selfie, clothes, room, food, pet, furniture, car, or random non-academic object), "
        "you MUST mark 'is_schoolwork': false.\n\n"
        "TASK 2: EXTRACTION (Only if it IS schoolwork)\n"
        "- Transcribe the exact question text completely.\n"
        "- Use standard LaTeX notation enclosed in $...$ for inline equations and $$...$$ for display equations. Never use raw unicode for math symbols.\n"
        "- Identify the subject (Physics, Chemistry, Mathematics, Biology, etc.).\n"
        "- Identify the topic / chapter title.\n"
        "- Identify options A, B, C, D if present.\n"
        "- Provide a brief diagram description if a diagram is included.\n\n"
        "Return strictly valid JSON with no markdown fences:\n"
        "{\n"
        '  "is_schoolwork": true,\n'
        '  "confidence": 0.95,\n'
        '  "rejection_reason": null,\n'
        '  "subject": "Physics",\n'
        '  "topic": "Thermodynamics",\n'
        '  "title": "Cyclic process P-V max temperature",\n'
        '  "question_text": "...",\n'
        '  "options": {"A": "...", "B": "...", "C": "...", "D": "..."},\n'
        '  "diagram_description": null\n'
        "}"
    )

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            vision_resp = await client.post(
                GROQ_URL,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                },
                json={
                    "model": VISION_MODEL,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": vision_prompt},
                                {"type": "image_url", "image_url": {"url": img_b64}},
                            ],
                        }
                    ],
                    "temperature": 0.2,
                    "max_tokens": 1500,
                },
            )
        
        if vision_resp.status_code != 200:
            raise HTTPException(status_code=vision_resp.status_code, detail=f"Groq Vision Error: {vision_resp.text}")

        vision_data = vision_resp.json()
        raw_vision_content = vision_data.get("choices", [{}])[0].get("message", {}).get("content", "")

        # Extract JSON
        json_match = re.search(r"\{[\s\S]*\}", raw_vision_content)
        if json_match:
            try:
                parsed_vision = json.loads(json_match.group())
            except Exception:
                parsed_vision = {"is_schoolwork": True, "question_text": raw_vision_content}
        else:
            parsed_vision = {"is_schoolwork": True, "question_text": raw_vision_content}

        # Check schoolwork validation
        if not parsed_vision.get("is_schoolwork", True):
            return {
                "is_schoolwork": False,
                "error": "This doesn't look like schoolwork. Snap a question from your book or notebook.",
                "reason": parsed_vision.get("rejection_reason"),
            }

        question_text = parsed_vision.get("question_text", "").strip()
        if not question_text:
            return {
                "is_schoolwork": False,
                "error": "This doesn't look like schoolwork. Snap a question from your book or notebook.",
            }

        subject = parsed_vision.get("subject", "General Science")
        topic = parsed_vision.get("topic", "Doubt Solution")
        title = parsed_vision.get("title", f"{subject} — {topic}")
        options = parsed_vision.get("options", {})
        diagram_desc = parsed_vision.get("diagram_description", "")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vision processing failed: {str(e)}")

    # Step 2: Generate Solution using openai/gpt-oss-120b
    options_formatted = ""
    if options and isinstance(options, dict):
        options_formatted = "\n".join([f"{k}) {v}" for k, v in options.items() if v])

    reasoning_prompt = f"""You are a master academic educator and tutor solving a student's question.
Provide a crystal-clear, pedagogically sound, and beautifully formatted solution.

Subject: {subject}
Topic: {topic}
Question:
{question_text}

{f"Options:\n{options_formatted}" if options_formatted else ""}
{f"Diagram Context: {diagram_desc}" if diagram_desc else ""}

STRICT FORMATTING REQUIREMENTS:
1. Use Markdown for structure:
   - Provide a clear bold topic header at the start (e.g. **{subject} — {topic}**).
   - Include a concise subtitle if relevant.
   - Include an **Important points:** section with bullet points highlighting key formulas and laws.
   - Walk through the step-by-step solution clearly without skipping essential intermediate steps.
   - End with a clearly boxed or highlighted final answer.
2. LATEX DELIMITERS:
   - Every single mathematical formula, expression, symbol, or equation MUST be formatted in LaTeX.
   - Use $...$ for inline math (e.g. $T = \\frac{{PV}}{{nR}}$, $\\Delta U = 0$, $P_1 V_1 = P_2 V_2$).
   - Use $$...$$ on its own line with a blank line before and after for standalone/display equations.
   - NEVER use unicode mathematical characters (no ∂, ∫, ∑, θ, π, α, β bare in text) — write $\\partial$, $\\int$, $\\sum$, $\\theta$, $\\pi$, $\\alpha$, $\\beta$.
   - NEVER use \\[...\\] or \\(...\\) delimiters.
3. Keep the tone encouraging, clear, and easy for students to follow.
"""

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            reasoning_resp = await client.post(
                GROQ_URL,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                },
                json={
                    "model": REASONING_MODEL,
                    "messages": [
                        {"role": "system", "content": "You are an elite STEM tutor who explains concepts with pristine mathematical clarity and perfect LaTeX notation."},
                        {"role": "user", "content": reasoning_prompt},
                    ],
                    "temperature": 0.4,
                    "max_tokens": 2500,
                },
            )

        if reasoning_resp.status_code != 200:
            raise HTTPException(status_code=reasoning_resp.status_code, detail=f"Reasoning Error: {reasoning_resp.text}")

        reasoning_data = reasoning_resp.json()
        solution_text = reasoning_data.get("choices", [{}])[0].get("message", {}).get("content", "")

        return {
            "is_schoolwork": True,
            "subject": subject,
            "topic": topic,
            "title": title,
            "question_text": question_text,
            "options": options,
            "solution": solution_text,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Solution generation failed: {str(e)}")


@app.post("/snap-solve/transcribe")
async def snap_transcribe(req: SnapTranscribeRequest):
    """
    Transcribes audio stream/recording using Deepgram Nova-3 for ultra-low latency.
    """
    if not req.audio_base64:
        raise HTTPException(status_code=400, detail="audio_base64 is required")

    try:
        # Strip header if present
        raw_b64 = req.audio_base64
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]
        
        audio_bytes = base64.b64decode(raw_b64)
        mime = req.mime_type or "audio/webm"

        async with httpx.AsyncClient(timeout=30.0) as client:
            dg_resp = await client.post(
                DEEPGRAM_URL,
                headers={
                    "Authorization": f"Token {DEEPGRAM_API_KEY}",
                    "Content-Type": mime,
                },
                content=audio_bytes,
            )

        if dg_resp.status_code != 200:
            raise HTTPException(status_code=dg_resp.status_code, detail=f"Deepgram Error: {dg_resp.text}")

        dg_data = dg_resp.json()
        transcript = (
            dg_data.get("results", {})
            .get("channels", [{}])[0]
            .get("alternatives", [{}])[0]
            .get("transcript", "")
        )

        return {"transcript": transcript.strip()}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@app.post("/snap-solve/voice-doubt")
async def snap_voice_doubt(req: SnapVoiceDoubtRequest):
    """
    Voice AI Tutor:
    1. Uses openai/gpt-oss-120b to understand the student's doubt in the context
       of the snapped question and solution.
    2. Synthesizes expressive speech with canopylabs/orpheus-v1-english on Groq.
    3. Returns text explanation + base64 audio stream for real-time playback.
    """
    if not req.user_doubt:
        raise HTTPException(status_code=400, detail="user_doubt is required")

    voice_name = req.voice if req.voice in ["autumn", "diana", "hannah", "austin", "daniel", "troy"] else "autumn"

    # Conversational system prompt for the Voice AI Tutor
    system_prompt = (
        "You are an empathetic, world-class personal AI voice tutor sitting with the student. "
        "The student is looking at a question and its solution and has asked a doubt.\n"
        "RULES FOR YOUR RESPONSE:\n"
        "1. Speak naturally, warmly, and encouragingly in conversational English.\n"
        "2. Keep your explanation concise and punchy (2 to 4 sentences), so it is enjoyable to listen to via voice.\n"
        "3. Address the exact confusion directly without generic filler.\n"
        "4. If you refer to a formula, state the intuitive relationship clearly.\n"
        "5. You may format key equations in $...$ for the on-screen transcript."
    )

    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]

    # Provide context
    context_msg = (
        f"Context for the problem:\n"
        f"Question: {req.question_text}\n\n"
        f"Reference Solution:\n{req.solution}"
    )
    messages.append({"role": "user", "content": context_msg})
    messages.append({"role": "assistant", "content": "I understand the question and the solution. How can I help you understand this better?"})

    # Add conversation history if any
    for h in req.history[-6:]:
        if h.get("role") and h.get("content"):
            messages.append({"role": h["role"], "content": h["content"]})

    # Student's current doubt
    messages.append({"role": "user", "content": req.user_doubt})

    # Generate reasoning
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            gpt_resp = await client.post(
                GROQ_URL,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                },
                json={
                    "model": REASONING_MODEL,
                    "messages": messages,
                    "temperature": 0.5,
                    "max_tokens": 400,
                },
            )

        if gpt_resp.status_code != 200:
            raise HTTPException(status_code=gpt_resp.status_code, detail=f"AI Tutor Error: {gpt_resp.text}")

        gpt_data = gpt_resp.json()
        tutor_text = gpt_data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

        # Clean text for speech synthesis
        speech_text = clean_for_speech(tutor_text)
        if not speech_text:
            speech_text = "Let me explain that for you."

        # Synthesize audio with canopylabs/orpheus-v1-english
        async with httpx.AsyncClient(timeout=60.0) as client:
            tts_resp = await client.post(
                GROQ_TTS_URL,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                },
                json={
                    "model": TTS_MODEL,
                    "input": speech_text[:1000],
                    "voice": voice_name,
                    "response_format": "wav",
                },
            )

        audio_b64 = ""
        if tts_resp.status_code == 200:
            audio_b64 = base64.b64encode(tts_resp.content).decode("utf-8")

        return {
            "text": tutor_text,
            "speech_text": speech_text,
            "audio_base64": audio_b64,
            "voice": voice_name,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Voice doubt processing failed: {str(e)}")


@app.post("/snap-solve/tts")
async def snap_tts(req: SnapTTSRequest):
    """
    Direct TTS generation using canopylabs/orpheus-v1-english on Groq.
    """
    if not req.text:
        raise HTTPException(status_code=400, detail="text is required")

    voice_name = req.voice if req.voice in ["autumn", "diana", "hannah", "austin", "daniel", "troy"] else "autumn"
    speech_text = clean_for_speech(req.text)

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            tts_resp = await client.post(
                GROQ_TTS_URL,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                },
                json={
                    "model": TTS_MODEL,
                    "input": speech_text[:1000],
                    "voice": voice_name,
                    "response_format": "wav",
                },
            )

        if tts_resp.status_code != 200:
            raise HTTPException(status_code=tts_resp.status_code, detail=f"TTS Error: {tts_resp.text}")

        audio_b64 = base64.b64encode(tts_resp.content).decode("utf-8")
        return {"audio_base64": audio_b64, "voice": voice_name}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}")

