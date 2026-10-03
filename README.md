# Rookie FastAPI Backend

Replaces the Next.js `/api/solution` route with a standalone async FastAPI service.  
Only the `/solution` endpoint is exposed. The `determine_answer` action has been removed — `correct_option` now arrives from the database.

---

## Setup

```bash
cd fastapi_backend

# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
copy .env.example .env        # Windows
# cp .env.example .env        # macOS / Linux
# then edit .env and set GROQ_API_KEY
```

---

## Run

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be live at `http://localhost:8000`.  
Interactive docs: `http://localhost:8000/docs`

---

## POST `/solution`

### Shared fields

| Field | Type | Notes |
|---|---|---|
| `action` | string | **required** — one of the actions below |
| `question_text` | string | question body (markdown / LaTeX ok) |
| `option_A` … `option_D` | string | answer options |
| `option_a_img` … `option_d_img` | string | image URLs for options (optional) |
| `correct_option` | string | correct letter from DB — e.g. `"C"` |
| `solution` | string | reference solution from DB |
| `question_id` | string | DB row ID (optional, for logging) |
| `exam_shift` | string | e.g. `"JEE Main 2026 … Evening Shift"` |
| `subject` | string | e.g. `"chemistry"` |
| `chapter` | string | e.g. `"Some Basic Concepts of Chemistry"` |
| `option_a_percent` … `option_d_percent` | float | student response distribution (optional) |
| `buddy_id` | string | AI buddy ID (optional) |
| `buddy_name` | string | AI buddy display name (optional) |
| `buddy_system_prompt` | string | full system prompt for the buddy (optional) |
| `model` | string | Groq model ID (default: `openai/gpt-oss-120b`) |
| `temperature` | float | default `0.3` |
| `max_tokens` | int | default `500` |

---

### Actions

#### `generate_solution`
Generates a conversational step-by-step solution.  
**Requires:** `question_text`, `option_A/B/C/D`, `solution`, `correct_option`  
**Returns:** `{ "solution": "...", "full_response": {...} }`

#### `explain_5yr`
Explains the solution using simple analogies, like talking to a child.  
**Requires:** `question_text`, `solution`  
**Returns:** `{ "explanation": "...", "full_response": {...} }`

#### `better_understanding`
Gives a simpler, more intuitive explanation focused on the core concept.  
**Requires:** `question_text`, `solution`  
**Returns:** `{ "explanation": "...", "full_response": {...} }`

#### `dig_deeper`
Creates a simpler MCQ based on the same underlying concept.  
**Requires:** `question_text`, `solution`  
**Returns:**
```json
{
  "mcq": {
    "question": "...",
    "options": ["...", "...", "...", "..."],
    "correctAnswer": "A",
    "explanation": "..."
  },
  "raw_response": "..."
}
```

---

## Example request

```bash
curl -X POST http://localhost:8000/solution \
  -H "Content-Type: application/json" \
  -d '{
    "action": "generate_solution",
    "question_text": "What is the number of atoms in 1.8 mg of water?",
    "option_A": "2×10⁻⁴×Nₐ",
    "option_B": "1.5×10⁻⁴×Nₐ",
    "option_C": "3×10⁻⁴×Nₐ",
    "option_D": "7×10⁻⁴×Nₐ",
    "correct_option": "C",
    "solution": "Moles of H₂O = 1.8e-3/18 = 1e-4 mol; 3 atoms per molecule → 3×10⁻⁴×Nₐ",
    "subject": "chemistry",
    "chapter": "Some Basic Concepts of Chemistry",
    "option_a_percent": 6.0,
    "option_b_percent": 11.0,
    "option_c_percent": 75.0,
    "option_d_percent": 9.0,
    "buddy_name": "Rahul",
    "buddy_system_prompt": "You are Rahul, a chill Delhi boy who explains JEE concepts..."
  }'
```
