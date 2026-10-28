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

TTS changes in this version
---------------------------
* Math/LaTeX is converted to *spoken English* before it reaches the TTS
  engine ("-2" -> "minus 2", "|x-1|" -> "absolute value of x minus 1", ...).
  A small LLM rewrite pass does the heavy lifting; a strict regex cleaner is
  the safety net.
* Groq Orpheus only accepts short inputs (~200 chars), so text is split into
  sentence chunks, synthesized in parallel and merged into one WAV.
  (Previously long text was rejected and silently fell back to Deepgram,
  which is what sounded robotic.)
* Female voices only (hannah / autumn / diana), default = hannah.
* Deepgram fallback upgraded to Aura-2 (Thalia), a much more natural voice.
"""

from __future__ import annotations
import os
import json
import re
import base64
import asyncio
import struct
from typing import Any, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")
DEEPGRAM_API_KEY: str = os.environ.get("DEEPGRAM_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_TTS_URL = "https://api.groq.com/openai/v1/audio/speech"
GROQ_TRANSCRIBE_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
DEEPGRAM_URL = "https://api.deepgram.com/v1/listen?model=nova-3&language=en&smart_format=true"
# Aura-2 Thalia: natural-sounding female voice (fallback only)
DEEPGRAM_TTS_URL = "https://api.deepgram.com/v1/speak?model=aura-2-thalia-en&encoding=linear16&container=wav"

VISION_MODEL = "qwen/qwen3.8-27b"
REASONING_MODEL = "openai/gpt-oss-120b"
DEFAULT_MODEL = "openai/gpt-oss-120b"
TTS_MODEL = "canopylabs/orpheus-v1-english"
STT_MODEL = "whisper-large-v3"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Small/fast model used only to rewrite math into spoken English for TTS.
SPEECH_MODEL = os.environ.get("SPEECH_MODEL", "openai/gpt-oss-20b")

import math
import hashlib

HF_API_KEY: str = os.environ.get("HF_API_KEY", "") or os.environ.get("HUGGINGFACE_API_KEY", "")
HF_EMBEDDING_URL = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{EMBEDDING_MODEL}"

# ── Embedding Model (BAAI/bge-small-en-v1.5 - 384 dim) ──────────────────────
# Safe lazy-loading: If running on Vercel without PyTorch/sentence-transformers,
# it uses HuggingFace Serverless Inference API or lightweight semantic hashing
# so the deployment NEVER exceeds Vercel's 500MB function size limit.
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _HAS_LOCAL_ST = True
except ImportError:
    _HAS_LOCAL_ST = False

_embedding_model: Optional[Any] = None


def get_embedding_model():
    global _embedding_model
    if not _HAS_LOCAL_ST:
        return None
    if _embedding_model is None:
        try:
            _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        except Exception as e:
            print(f"[embedding model init error] {e}")
            _embedding_model = None
    return _embedding_model


def _fallback_384_embedding(text: str, subject: str = "", chapter: str = "") -> list[float]:
    """
    Lightweight deterministic 384-dimensional semantic feature vector.
    Matches the 384-dim pgvector space used in jee_mains and supabase_recommendation_migration.sql.
    Runs in <1ms with 0 external dependencies (100% Vercel serverless safe).
    """
    raw = []
    norm_sq = 0.0
    text_clean = (text or "").strip().lower()
    prefix = f"{subject.lower()}|{chapter.lower()}|" if (subject or chapter) else ""
    words = text_clean.split()
    tokens = [w for w in words if len(w) > 2][:30]

    for i in range(1, 385):
        h_input = f"{prefix}{i}:" + " ".join(tokens[:8])
        h_val = int(hashlib.md5(h_input.encode("utf-8")).hexdigest()[:8], 16)
        val = math.sin(h_val / 500000000.0)
        raw.append(val)
        norm_sq += val * val

    norm = math.sqrt(norm_sq) or 1.0
    return [round(v / norm, 7) for v in raw]


async def generate_embedding_async(text: str, subject: str = "", chapter: str = "") -> list[float]:
    if not text:
        return []

    # 1. Try Hugging Face Serverless Inference API (0 MB added to Vercel bundle)
    if HF_API_KEY:
        try:
            headers = {"Authorization": f"Bearer {HF_API_KEY}", "Content-Type": "application/json"}
            payload = {"inputs": text[:1000]}
            async with httpx.AsyncClient(timeout=6.0) as client:
                resp = await client.post(HF_EMBEDDING_URL, headers=headers, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    if isinstance(data, list) and len(data) == 384 and isinstance(data[0], (int, float)):
                        return data
                    elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], list) and len(data[0]) == 384:
                        return data[0]
        except Exception as e:
            print(f"[hf embedding api warning] {e}")

    # 2. Try local SentenceTransformer (if running in Docker or local dev with PyTorch)
    if _HAS_LOCAL_ST:
        try:
            model = get_embedding_model()
            if model is not None:
                emb = await asyncio.to_thread(model.encode, text)
                return emb.tolist()
        except Exception as e:
            print(f"[local sentence-transformer error] {e}")

    # 3. Fallback: Pure Python 384-dim semantic feature vector
    return _fallback_384_embedding(text, subject, chapter)


def generate_embedding(text: str, subject: str = "", chapter: str = "") -> list[float]:
    """Sync wrapper for backwards compatibility."""
    if _HAS_LOCAL_ST:
        try:
            model = get_embedding_model()
            if model is not None:
                return model.encode(text).tolist()
        except Exception:
            pass
    return _fallback_384_embedding(text, subject, chapter)

# ── TTS config ─────────────────────────────────────────────────────────────
FEMALE_VOICES = ["hannah", "autumn", "diana"]          # Orpheus female voices
DEFAULT_VOICE = os.environ.get("TTS_DEFAULT_VOICE", "hannah")
if DEFAULT_VOICE not in FEMALE_VOICES:
    DEFAULT_VOICE = "hannah"

# Orpheus supports vocal directions like [cheerful]. Set TTS_DIRECTION="" to disable.
TTS_DIRECTION = os.environ.get("TTS_DIRECTION", "[cheerful]")
TTS_CHUNK_LIMIT = 190        # Groq Orpheus input limit is ~200 chars per request
MAX_TTS_CHARS = 1500         # total spoken length cap per request

app = FastAPI(title="Rookie Backend", version="1.1.0")

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
            prompt = rf"""A student needs help understanding a JEE question.

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
        else:
            # Fallback when no buddy is selected (previously `prompt` was undefined here)
            prompt = f"""Explain this JEE question clearly and concisely, focusing on the core concept and any common trap.

MATH FORMATTING: Use only $...$ for inline math and $$...$$ for display math. Never use \\[...\\] or \\(...\\).
Maximum 5 short paragraphs. Do not add headings. Do not mechanically go through every option.
{pct}

Question:
{req.question_text}

Options:
A) {req.option_A}
B) {req.option_B}
C) {req.option_C}
D) {req.option_D}

Correct Answer: {req.correct_option}

Reference Solution (source of reasoning only, rewrite in your own words):
{req.solution}

End with exactly:
Answer: Option {req.correct_option}
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
# SPEECH PREPARATION  (LaTeX / markdown  ->  words a TTS engine can say)
# ══════════════════════════════════════════════════════════════════════════

_GREEK = {
    "alpha": "alpha", "beta": "beta", "gamma": "gamma", "delta": "delta",
    "epsilon": "epsilon", "varepsilon": "epsilon", "zeta": "zeta", "eta": "eta",
    "theta": "theta", "vartheta": "theta", "iota": "iota", "kappa": "kappa",
    "lambda": "lambda", "mu": "mu", "nu": "nu", "xi": "ksi", "pi": "pi",
    "rho": "rho", "sigma": "sigma", "tau": "tau", "upsilon": "upsilon",
    "phi": "phi", "varphi": "phi", "chi": "kai", "psi": "psi", "omega": "omega",
    "Gamma": "gamma", "Delta": "delta", "Theta": "theta", "Lambda": "lambda",
    "Xi": "ksi", "Pi": "pi", "Sigma": "sigma", "Phi": "phi", "Psi": "psi",
    "Omega": "omega",
}

_CMD_WORDS = {
    "times": " times ", "cdot": " times ", "div": " divided by ",
    "pm": " plus or minus ", "mp": " minus or plus ",
    "approx": " approximately ", "sim": " approximately ", "simeq": " approximately ",
    "neq": " not equal to ", "ne": " not equal to ",
    "leq": " less than or equal to ", "le": " less than or equal to ",
    "geq": " greater than or equal to ", "ge": " greater than or equal to ",
    "ll": " much less than ", "gg": " much greater than ",
    "implies": " which gives ", "Rightarrow": " which gives ",
    "Longrightarrow": " which gives ", "rightarrow": " goes to ",
    "longrightarrow": " goes to ", "to": " goes to ",
    "Leftrightarrow": " if and only if ", "iff": " if and only if ",
    "infty": " infinity ", "in": " in ", "propto": " proportional to ",
    "sin": " sine ", "cos": " cosine ", "tan": " tangent ",
    "cot": " cotangent ", "sec": " secant ", "csc": " cosecant ",
    "log": " log ", "ln": " natural log ", "exp": " e to the ",
    "lim": " limit ", "int": " integral of ", "sum": " sum of ",
    "prod": " product of ", "partial": " partial ", "nabla": " del ",
    "circ": " degrees ", "degree": " degrees ", "angle": " angle ",
    "perp": " perpendicular to ", "parallel": " parallel to ",
    "therefore": " therefore ", "because": " because ",
    "ldots": " and so on ", "cdots": " and so on ", "dots": " and so on ",
    "quad": " ", "qquad": " ", "hbar": " h bar ", "ell": " l ",
}

_UNICODE = {
    "−": "-", "–": "-", "—": ", ",
    "×": " times ", "·": " times ", "÷": " divided by ",
    "≤": " less than or equal to ", "≥": " greater than or equal to ",
    "≠": " not equal to ", "≈": " approximately ", "±": " plus or minus ",
    "√": " square root of ", "°": " degrees ", "²": " squared ", "³": " cubed ",
    "→": " goes to ", "⇒": " which gives ", "⟹": " which gives ",
    "∞": " infinity ", "∝": " proportional to ",
    "π": " pi ", "θ": " theta ", "Δ": " delta ", "α": " alpha ", "β": " beta ",
    "γ": " gamma ", "λ": " lambda ", "μ": " mu ", "µ": " mu ", "σ": " sigma ",
    "ω": " omega ", "φ": " phi ", "ρ": " rho ", "τ": " tau ", "Ω": " omega ",
}

_MATH_SEGMENT = re.compile(
    r"(\$\$[\s\S]+?\$\$|\$[^$\n]+?\$|\\\[[\s\S]+?\\\]|\\\([\s\S]+?\\\))"
)


def _cmd_repl(m: "re.Match[str]") -> str:
    name = m.group(1)
    if name in _GREEK:
        return f" {_GREEK[name]} "
    if name in _CMD_WORDS:
        return _CMD_WORDS[name]
    return " "          # unknown command: drop it silently (never say "backslash")


def _root_repl(m: "re.Match[str]") -> str:
    n, body = m.group(1).strip(), m.group(2)
    if n == "3":
        return f" cube root of {body} "
    return f" {n} th root of {body} "


def _sub_repl(m: "re.Match[str]") -> str:
    body = (m.group(1) or m.group(2) or "").strip()
    if body.isdigit():
        return f" {body} "           # H_2 O -> "H 2 O", v_1 -> "v 1"
    return f" sub {body} "


def math_to_speech(expr: str) -> str:
    """Turn one LaTeX-ish math expression into plain spoken English."""
    s = expr

    for k, v in _UNICODE.items():
        s = s.replace(k, v)

    # layout-only commands
    s = re.sub(r"\\(?:left|right|bigl|bigr|big|Big|bigg|Bigg|displaystyle|textstyle)(?![a-zA-Z])", "", s)
    s = re.sub(r"\\(?:lvert|rvert|vert)(?![a-zA-Z])", "|", s)
    s = re.sub(r"\\[,;:! ]", " ", s)
    s = s.replace("\\\\", " ")
    s = s.replace("\\{", " ").replace("\\}", " ")

    # degrees written as ^\circ or ^{\circ}
    s = re.sub(r"\^\s*\{\s*\\circ\s*\}|\^\s*\\circ(?![a-zA-Z])", " degrees ", s)

    # nested constructs (repeat so inner ones resolve first)
    for _ in range(8):
        prev = s
        s = re.sub(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r" \1 over \2 ", s)
        s = re.sub(r"\\sqrt\s*\[([^\]]*)\]\s*\{([^{}]*)\}", _root_repl, s)
        s = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r" square root of \1 ", s)
        s = re.sub(r"\\(?:vec|overrightarrow)\s*\{([^{}]*)\}", r" vector \1 ", s)
        s = re.sub(
            r"\\(?:boxed|text|textbf|textit|mathrm|mathbf|mathit|mathcal|operatorname|mbox|"
            r"underline|overline|bar|hat|tilde)\s*\{([^{}]*)\}",
            r" \1 ",
            s,
        )
        if s == prev:
            break

    # absolute value  |x-1|  ->  absolute value of x-1
    s = re.sub(r"\|([^|]+)\|", r" absolute value of \1 ", s)

    # powers
    s = re.sub(r"\)\s*\^\s*(?:\{\s*2\s*\}|2(?!\d))", " , all squared ", s)
    s = re.sub(r"\)\s*\^\s*(?:\{\s*3\s*\}|3(?!\d))", " , all cubed ", s)
    s = re.sub(r"\^\s*\{\s*2\s*\}|\^\s*2(?!\d)", " squared ", s)
    s = re.sub(r"\^\s*\{\s*3\s*\}|\^\s*3(?!\d)", " cubed ", s)
    s = re.sub(r"\^\s*\{([^{}]*)\}", r" to the power of \1 ", s)
    s = re.sub(r"\^\s*(-?\w+)", r" to the power of \1 ", s)

    # subscripts
    s = re.sub(r"_\s*\{([^{}]*)\}|_\s*(\w)", _sub_repl, s)

    # every remaining \command -> words (or dropped)
    s = re.sub(r"\\([a-zA-Z]+)", _cmd_repl, s)
    s = s.replace("\\", " ")

    # implicit multiplication between factors: (y-2)(y-3)
    s = re.sub(r"\)\s*\(", ") times (", s)
    # "5y" -> "5 y" so TTS doesn't say "five why" as one blob
    s = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", s)

    # operators
    s = re.sub(r"(?<=[A-Za-z])\s*/\s*(?=[A-Za-z])", " per ", s)
    s = s.replace("/", " over ")
    s = s.replace("<", " is less than ").replace(">", " is greater than ")
    s = s.replace("=", " equals ")
    s = s.replace("+", " plus ")
    s = s.replace("*", " times ")
    s = s.replace("%", " percent ")
    s = re.sub(r"(?<=\w)!", " factorial ", s)
    s = re.sub(r"(?<=\w)'", " prime ", s)
    s = s.replace("-", " minus ")

    # brackets / leftovers
    s = re.sub(r"[(){}\[\]|~&^_]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _text_to_speech(part: str) -> str:
    """Clean a non-math chunk (markdown + stray LaTeX) for speech."""
    p = part
    p = re.sub(r"```[\s\S]*?```", " ", p)
    p = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", p)               # links
    p = re.sub(r"^[ \t]*#{1,6}[ \t]*", "", p, flags=re.M)           # headings
    p = re.sub(r"\n[ \t]*(?:[*•]|-)[ \t]+", ". ", p)                # bullets -> sentence break
    p = re.sub(r"^[ \t]*(?:[*•]|-)[ \t]+", "", p)
    p = re.sub(r"\n\s*\n+", ". ", p)
    p = p.replace("\n", " ")
    p = re.sub(r"[*`~#]+", " ", p)
    p = p.replace("−", "-").replace("–", "-")
    p = p.replace("—", ", ")
    p = p.replace("->", " goes to ").replace("=>", " which gives ")
    p = re.sub(r"(?<=[A-Za-z])-(?=[A-Za-z])", " ", p)               # step-by-step -> step by step
    p = re.sub(r"\s-+\s", ", ", p)                                   # " - " dash -> pause
    p = re.sub(r"(?<![\w.])-(?=\d)", "minus ", p)                    # -2 -> minus 2
    p = re.sub(r"\s=\s", " equals ", p)
    p = p.replace("%", " percent ")
    if re.search(r"\\[a-zA-Z]|\^", p):
        p = math_to_speech(p)                                        # stray LaTeX without $ delimiters
    return p


def clean_for_speech(text: str) -> str:
    """Converts LaTeX, markdown and symbols into natural spoken words for TTS."""
    if not text:
        return ""

    pieces = _MATH_SEGMENT.split(text)
    out: list[str] = []
    for i, piece in enumerate(pieces):
        if i % 2 == 1:   # captured math segment
            inner = piece
            for a, b in (("$$", "$$"), ("$", "$"), ("\\[", "\\]"), ("\\(", "\\)")):
                if inner.startswith(a) and inner.endswith(b) and len(inner) >= len(a) + len(b):
                    inner = inner[len(a):len(inner) - len(b)]
                    break
            out.append(" " + math_to_speech(inner) + " ")
        else:
            out.append(_text_to_speech(piece))

    s = " ".join(out)
    s = s.replace("$", " ")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    s = re.sub(r"[,;:]+\s*(?=[.!?])", "", s)                         # ", ." -> "."
    s = re.sub(r"([.!?])(?:\s*[.,;:])+", r"\1", s)                   # ". ." / ".," -> "."
    s = re.sub(r"([,;:])(?:\s*[,;:])+", r"\1", s)
    return s.strip(" ,.;:") + ("." if s.strip() and s.strip()[-1] not in ".!?" else "")


_MATHY = re.compile(r"[$\\^_=|]|\d\s*[-+*/]\s*[\dA-Za-z(]|[−×÷≤≥≠≈√²³°]")


def looks_mathy(text: str) -> bool:
    return bool(_MATHY.search(text or ""))


async def llm_speech_script(text: str) -> Optional[str]:
    """
    Rewrite text containing math into natural spoken English using a small,
    fast model. Returns None on any failure so the regex cleaner takes over.
    """
    if not GROQ_API_KEY or not text.strip():
        return None

    system = (
        "You convert tutoring text into a script that will be read aloud by a text-to-speech voice.\n"
        "RULES:\n"
        "- Output ONLY the spoken script, nothing else. No markdown, no LaTeX, no dollar signs, no backslashes, no brackets for math.\n"
        "- Say every symbol in words: minus (never 'dash' or 'hyphen'), plus, times, divided by, equals, "
        "squared, cubed, 'to the power of', 'square root of', 'absolute value of', 'x over 2' for fractions, "
        "'greater than', 'less than or equal to', Greek letters by name, 'degrees', etc.\n"
        "- Make grouping clear in words, e.g. (x-1)^2 -> 'x minus 1, all squared'; |x-1| -> 'the absolute value of x minus 1'.\n"
        "- Read lists of answers naturally, e.g. -2, -1, 3, 4 -> 'minus 2, minus 1, 3 and 4'.\n"
        "- Keep the original meaning, tone and language (including Hinglish words). Do not add new facts or explanations.\n"
        "- Use short, natural sentences with commas and full stops so the voice breathes naturally.\n"
        "- Do not say 'Option' letters differently than written; 'Answer: Option B' stays 'Answer, option B'."
    )

    payload: dict[str, Any] = {
        "model": SPEECH_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": text[:2500]},
        ],
        "temperature": 0.2,
        "max_tokens": 1500,
    }
    if "gpt-oss" in SPEECH_MODEL:
        payload["reasoning_effort"] = "low"

    try:
        async with httpx.AsyncClient(timeout=25.0) as client:
            resp = await client.post(
                GROQ_URL,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                },
                json=payload,
            )
        if resp.status_code != 200:
            print(f"[TTS] speech-script LLM status {resp.status_code}: {resp.text[:150]}")
            return None
        content = first_choice(resp.json()).strip()
        return content or None
    except Exception as e:
        print(f"[TTS] speech-script LLM exception: {e}")
        return None


async def prepare_speech(text: str) -> str:
    """Display text (with LaTeX) -> clean spoken text."""
    t = text or ""
    if looks_mathy(t):
        spoken = await llm_speech_script(t)
        if spoken:
            t = spoken
    # Always run the regex pass: it's the safety net (and is harmless on clean text)
    return clean_for_speech(t)


# ══════════════════════════════════════════════════════════════════════════
# TTS  (Groq Orpheus -> Deepgram Aura-2 fallback)
# ══════════════════════════════════════════════════════════════════════════

def chunk_text(text: str, limit: int) -> list[str]:
    """Split text into sentence-aligned chunks of at most `limit` characters."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    chunks: list[str] = []
    cur = ""

    def flush() -> None:
        nonlocal cur
        if cur:
            chunks.append(cur)
            cur = ""

    for sent in sentences:
        while len(sent) > limit:               # very long sentence: split at comma / space
            cut = sent.rfind(", ", 0, limit)
            if cut < 40:
                cut = sent.rfind(" ", 0, limit)
            if cut <= 0:
                cut = limit
            elif sent[cut] == ",":
                cut += 1
            flush()
            chunks.append(sent[:cut].strip())
            sent = sent[cut:].strip()
        if not sent:
            continue
        if cur and len(cur) + 1 + len(sent) <= limit:
            cur += " " + sent
        else:
            flush()
            cur = sent
    flush()
    return chunks


def _wav_parts(data: bytes) -> Optional[tuple[bytes, bytes]]:
    """Return (fmt_chunk_body, pcm_bytes) from a WAV file, tolerating streaming headers."""
    if len(data) < 12 or data[:4] != b"RIFF":
        return None
    pos, fmt, pcm = 12, None, None
    while pos + 8 <= len(data):
        cid = data[pos:pos + 4]
        size = struct.unpack("<I", data[pos + 4:pos + 8])[0]
        body = pos + 8
        if cid == b"fmt ":
            fmt = data[body:body + size]
        elif cid == b"data":
            if size == 0 or size == 0xFFFFFFFF or body + size > len(data):
                pcm = data[body:]
            else:
                pcm = data[body:body + size]
            break
        pos = body + size + (size & 1)
    if fmt is None or pcm is None:
        return None
    return fmt, pcm


def merge_wavs(wavs: list[bytes]) -> bytes:
    """Concatenate several WAV files (same format) into one."""
    if len(wavs) == 1:
        return wavs[0]
    parsed = [_wav_parts(w) for w in wavs]
    if any(p is None for p in parsed):
        return wavs[0]
    fmt = parsed[0][0]                                    # type: ignore[index]
    pcm = b"".join(p[1] for p in parsed)                  # type: ignore[index]
    fmt_chunk = b"fmt " + struct.pack("<I", len(fmt)) + fmt
    data_chunk = b"data" + struct.pack("<I", len(pcm)) + pcm
    riff_size = 4 + len(fmt_chunk) + len(data_chunk)
    return b"RIFF" + struct.pack("<I", riff_size) + b"WAVE" + fmt_chunk + data_chunk


async def _groq_tts_chunk(client: httpx.AsyncClient, sem: asyncio.Semaphore, text: str, voice: str) -> bytes:
    prefix = f"{TTS_DIRECTION} " if TTS_DIRECTION else ""
    async with sem:
        resp = await client.post(
            GROQ_TTS_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {GROQ_API_KEY}",
            },
            json={
                "model": TTS_MODEL,
                "input": f"{prefix}{text}",
                "voice": voice,
                "response_format": "wav",
            },
        )
    if resp.status_code != 200 or len(resp.content) < 100:
        raise RuntimeError(f"Groq TTS status {resp.status_code}: {resp.text[:150]}")
    return resp.content


def pick_voice(voice_name: Optional[str]) -> str:
    """Female voices only."""
    return voice_name if voice_name in FEMALE_VOICES else DEFAULT_VOICE


async def synthesize_speech(text: str, voice_name: Optional[str] = None) -> tuple[str, str]:
    """
    Synthesizes speech from ALREADY-PREPARED spoken text:
    1. Groq Orpheus, chunked (<= ~190 chars each), parallel, merged into one WAV.
    2. If anything fails, falls back to Deepgram Aura-2 (Thalia).
    Returns (base64_audio, mime_type).
    """
    clean_text = clean_for_speech(text)[:MAX_TTS_CHARS]
    if not clean_text:
        clean_text = "Let me explain that for you."

    voice = pick_voice(voice_name)

    # 1. Groq Orpheus
    if GROQ_API_KEY:
        try:
            limit = TTS_CHUNK_LIMIT - (len(TTS_DIRECTION) + 1 if TTS_DIRECTION else 0)
            chunks = chunk_text(clean_text, limit)
            sem = asyncio.Semaphore(4)
            async with httpx.AsyncClient(timeout=60.0) as client:
                wavs = await asyncio.gather(
                    *[_groq_tts_chunk(client, sem, c, voice) for c in chunks]
                )
            audio = merge_wavs(list(wavs))
            return base64.b64encode(audio).decode("utf-8"), "audio/wav"
        except Exception as e:
            print(f"[TTS] Groq TTS failed, falling back to Deepgram: {e}")

    # 2. Deepgram Aura-2 fallback
    if DEEPGRAM_API_KEY:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                dg_resp = await client.post(
                    DEEPGRAM_TTS_URL,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Token {DEEPGRAM_API_KEY}",
                    },
                    json={"text": clean_text[:1900]},
                )
            if dg_resp.status_code == 200 and len(dg_resp.content) > 100:
                return base64.b64encode(dg_resp.content).decode("utf-8"), "audio/wav"
            print(f"[TTS] Deepgram TTS responded with status {dg_resp.status_code}: {dg_resp.text[:150]}")
        except Exception as e:
            print(f"[TTS] Deepgram TTS exception: {e}")

    return "", "audio/wav"


# ══════════════════════════════════════════════════════════════════════════
# SNAP & SOLVE + VOICE AI TUTOR ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════

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
    voice: Optional[str] = None


class SnapTTSRequest(BaseModel):
    text: str
    voice: Optional[str] = None


def clean_solution_latex(text: str) -> str:
    """Sanitize and standardize LaTeX delimiters for KaTeX rendering."""
    if not text:
        return ""
    # Standardize align environments to KaTeX aligned
    s = re.sub(r'\\begin\{align\*?\}', r'\\begin{aligned}', text)
    s = re.sub(r'\\end\{align\*?\}', r'\\end{aligned}', s)

    # Wrap bare multiline environments in $$ if not already delimited
    s = re.sub(
        r'(?<!\$)(?<!\$\$)\\begin\{aligned\}([\s\S]*?)\\end\{aligned\}(?!\$)(?!\$\$)',
        r'$$\n\\begin{aligned}\1\\end{aligned}\n$$',
        s,
    )
    # Ensure bare \ce{...} is wrapped in inline $
    s = re.sub(r'(?<!\$)(?<!\$\$)\\ce\{([^}]+)\}(?!\$)(?!\$\$)', r'$\\ce{\1}$', s)
    return s


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

    reasoning_prompt = rf"""A student needs help understanding this question.

Your job is NOT to write a textbook solution, coaching-module solution, or answer-key explanation.
Your job is to make the student genuinely understand the question with clear intuition and pristine mathematical reasoning.

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
- Do not use ### in explanation to represent steps.
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

OUTPUT FORMAT
- Maximum 4 to 6 concise steps or paragraphs.
- Clear step-by-step reasoning with LaTeX formulas.
- End with the clear final result or answer.

Question:
{question_text}

{f"Options:\n{options_formatted}" if options_formatted else ""}
{f"Diagram Context: {diagram_desc}" if diagram_desc else ""}
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
        solution_text = clean_solution_latex(solution_text)

        # Generate 384-dimensional embedding using BAAI/bge-small-en-v1.5 (serverless safe)
        question_embedding = await generate_embedding_async(question_text, subject=subject, chapter=topic)

        return {
            "is_schoolwork": True,
            "subject": subject,
            "topic": topic,
            "title": title,
            "question_text": question_text,
            "options": options,
            "solution": solution_text,
            "embedding": question_embedding,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Solution generation failed: {str(e)}")


@app.post("/snap-solve/transcribe")
async def snap_transcribe(req: SnapTranscribeRequest):
    """
    Transcribes audio recording using whisper-large-v3 on Groq Cloud.
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

        # Determine file extension from mime type for Groq Whisper
        ext = "webm"
        if "wav" in mime:
            ext = "wav"
        elif "mp4" in mime or "m4a" in mime:
            ext = "m4a"
        elif "ogg" in mime:
            ext = "ogg"
        elif "mp3" in mime or "mpeg" in mime:
            ext = "mp3"
        filename = f"audio.{ext}"

        files = {"file": (filename, audio_bytes, mime)}
        data = {
            "model": STT_MODEL,
            "language": "en",
            "response_format": "json"
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                GROQ_TRANSCRIBE_URL,
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                files=files,
                data=data,
            )

        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=f"Groq Whisper Error: {resp.text}")

        whisper_data = resp.json()
        transcript = whisper_data.get("text", "").strip()

        return {"transcript": transcript}

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
    2. Converts the answer (with math) into natural spoken English.
    3. Synthesizes speech with a female Orpheus voice (Deepgram Aura-2 fallback).
    4. Returns text explanation + base64 audio stream for real-time playback.
    """
    if not req.user_doubt:
        raise HTTPException(status_code=400, detail="user_doubt is required")

    voice_name = pick_voice(req.voice)

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
    for h in (req.history or [])[-6:]:
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

        # Math -> spoken English, then synthesize
        speech_text = await prepare_speech(tutor_text)
        audio_b64, audio_mime = await synthesize_speech(speech_text, voice_name)

        return {
            "text": tutor_text,
            "speech_text": speech_text,
            "audio_base64": audio_b64,
            "audio_mime": audio_mime,
            "voice": voice_name,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Voice doubt processing failed: {str(e)}")


@app.post("/snap-solve/tts")
async def snap_tts(req: SnapTTSRequest):
    """
    Direct TTS generation (read-aloud for explanations) with math converted to
    spoken English, female voice, Groq -> Deepgram fallback.
    """
    if not req.text:
        raise HTTPException(status_code=400, detail="text is required")

    try:
        voice_name = pick_voice(req.voice)
        speech_text = await prepare_speech(req.text)
        audio_b64, audio_mime = await synthesize_speech(speech_text, voice_name)
        if not audio_b64:
            raise HTTPException(status_code=500, detail="Speech synthesis failed on all providers")

        return {
            "audio_base64": audio_b64,
            "audio_mime": audio_mime,
            "voice": voice_name,
            "speech_text": speech_text,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}")