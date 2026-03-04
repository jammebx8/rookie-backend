import { NextResponse } from 'next/server';
import axios from 'axios';

// Place this file at: app/api/notes/route.ts

const CORS_HEADERS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type, Authorization',
};

const VISION_MODEL = 'meta-llama/llama-4-scout-17b-16e-instruct';
const TEXT_MODEL = 'llama-3.3-70b-versatile';

export async function OPTIONS() {
  return new NextResponse(null, { status: 204, headers: CORS_HEADERS });
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { action, images, extracted_text } = body || {};

    // ── Shared Groq caller (text) ─────────────────────────────────────────────
    const callGroqText = async (
      messages: { role: string; content: string }[],
      opts: { max_tokens?: number; temperature?: number } = {}
    ) => {
      const res = await axios.post(
        'https://api.groq.com/openai/v1/chat/completions',
        {
          model: TEXT_MODEL,
          messages,
          temperature: opts.temperature ?? 0.5,
          max_tokens: opts.max_tokens ?? 2000,
        },
        {
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${process.env.GROQ_API_KEY}`,
          },
          timeout: 120_000,
        }
      );
      return res.data;
    };

    // ── Vision caller (Llama 4 Scout) ─────────────────────────────────────────
    const callGroqVision = async (
      imageBase64Array: { data: string; mediaType: string }[],
      prompt: string
    ) => {
      // Build multi-image content
      const imageContent = imageBase64Array.map(({ data, mediaType }) => ({
        type: 'image_url',
        image_url: {
          url: `data:${mediaType};base64,${data}`,
        },
      }));

      const res = await axios.post(
        'https://api.groq.com/openai/v1/chat/completions',
        {
          model: VISION_MODEL,
          messages: [
            {
              role: 'user',
              content: [
                ...imageContent,
                { type: 'text', text: prompt },
              ],
            },
          ],
          temperature: 0.2,
          max_tokens: 4000,
        },
        {
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${process.env.GROQ_API_KEY}`,
          },
          timeout: 120_000,
        }
      );
      return res.data;
    };

    // ════════════════════════════════════════════════════════════════════════════
    // ACTION: extract_notes
    // Takes base64 images → extracts all text content using Llama 4 Scout
    // ════════════════════════════════════════════════════════════════════════════
    if (action === 'extract_notes') {
      if (!images || !Array.isArray(images) || images.length === 0) {
        return new NextResponse(
          JSON.stringify({ error: 'images array is required' }),
          { status: 400, headers: { ...CORS_HEADERS, 'Content-Type': 'application/json' } }
        );
      }

      const extractPrompt = `You are an expert at reading handwritten and printed notes, textbook pages, and study material.

Your task: Extract ALL text from the provided image(s) of study notes. Be thorough and accurate.

Instructions:
- Extract every piece of text, including headings, subheadings, body text, equations, diagrams descriptions, and margin notes
- Preserve the logical structure (use headings, bullet points, numbered lists as appropriate)
- For mathematical equations, write them in plain readable form (e.g., "v² = u² + 2as")
- If there are multiple pages/images, combine them into one coherent document
- Do NOT add any commentary - just extract what is written
- If something is unclear, make a best guess and mark it with [?]

Return the extracted text in clean, well-structured markdown format.`;

      const groqData = await callGroqVision(images, extractPrompt);
      const extractedText = groqData.choices?.[0]?.message?.content || '';

      return new NextResponse(
        JSON.stringify({ extracted_text: extractedText }),
        { status: 200, headers: { ...CORS_HEADERS, 'Content-Type': 'application/json' } }
      );
    }

    // ════════════════════════════════════════════════════════════════════════════
    // ACTION: summarize_notes
    // Takes extracted text → generates a structured summary
    // ════════════════════════════════════════════════════════════════════════════
    else if (action === 'summarize_notes') {
      if (!extracted_text) {
        return new NextResponse(
          JSON.stringify({ error: 'extracted_text is required' }),
          { status: 400, headers: { ...CORS_HEADERS, 'Content-Type': 'application/json' } }
        );
      }

      const summarizePrompt = `You are an expert JEE/NEET tutor. A student has shared their study notes. Create a concise, well-structured summary that helps the student revise quickly.

Notes Content:
${extracted_text}

Create a summary with:
1. **Topic Title** - What subject/chapter is this about?
2. **Key Concepts** - The most important ideas (bullet points, max 8)
3. **Important Formulas** - List key formulas clearly (write equations in plain form like: F = ma)
4. **Quick Revision Points** - 3-5 things to remember for exams
5. **Difficulty Level** - Easy / Medium / Hard

Keep the summary focused, clear, and exam-oriented. Use emojis sparingly for visual scanning (e.g., 📌 for key points, ⚡ for formulas).`;

      const groqData = await callGroqText(
        [{ role: 'user', content: summarizePrompt }],
        { max_tokens: 1500, temperature: 0.4 }
      );
      const summary = groqData.choices?.[0]?.message?.content || '';

      return new NextResponse(
        JSON.stringify({ summary }),
        { status: 200, headers: { ...CORS_HEADERS, 'Content-Type': 'application/json' } }
      );
    }

    // ════════════════════════════════════════════════════════════════════════════
    // ACTION: generate_mcqs
    // Takes extracted text → generates MCQ quiz
    // ════════════════════════════════════════════════════════════════════════════
    else if (action === 'generate_mcqs') {
      if (!extracted_text) {
        return new NextResponse(
          JSON.stringify({ error: 'extracted_text is required' }),
          { status: 400, headers: { ...CORS_HEADERS, 'Content-Type': 'application/json' } }
        );
      }

      const mcqPrompt = `You are an expert JEE/NEET exam question setter. Based on the following study notes, create 5 high-quality MCQ questions to test student understanding.

Notes Content:
${extracted_text}

Rules:
- Create exactly 5 MCQs
- Each question should test understanding, not just memory
- Questions should range from easy to medium to hard
- All 4 options should be plausible (no obviously wrong options)
- Include a clear explanation for the correct answer

Respond ONLY with a valid JSON array (no markdown, no extra text):
[
  {
    "id": 1,
    "question": "question text here",
    "options": ["Option A text", "Option B text", "Option C text", "Option D text"],
    "correctAnswer": 0,
    "explanation": "explanation of why this answer is correct",
    "difficulty": "easy"
  }
]

Note: correctAnswer is the 0-based index (0=A, 1=B, 2=C, 3=D). difficulty is one of: easy, medium, hard.`;

      const groqData = await callGroqText(
        [{ role: 'user', content: mcqPrompt }],
        { max_tokens: 2000, temperature: 0.6 }
      );
      const aiResponse = groqData.choices?.[0]?.message?.content || '';

      // Parse JSON safely
      const jsonMatch = aiResponse.match(/\[[\s\S]*\]/);
      if (!jsonMatch) {
        return new NextResponse(
          JSON.stringify({ error: 'Could not parse MCQ JSON', raw: aiResponse }),
          { status: 500, headers: { ...CORS_HEADERS, 'Content-Type': 'application/json' } }
        );
      }

      const mcqs = JSON.parse(jsonMatch[0]);

      return new NextResponse(
        JSON.stringify({ mcqs }),
        { status: 200, headers: { ...CORS_HEADERS, 'Content-Type': 'application/json' } }
      );
    }

    // ════════════════════════════════════════════════════════════════════════════
    // Unknown action
    // ════════════════════════════════════════════════════════════════════════════
    else {
      return new NextResponse(
        JSON.stringify({ error: `Unknown action: ${action}` }),
        { status: 400, headers: { ...CORS_HEADERS, 'Content-Type': 'application/json' } }
      );
    }

  } catch (err: any) {
    console.error('notes route error:', err?.response?.data || err?.message || err);
    const status = err?.response?.status || 500;
    const details = err?.response?.data || err?.message || 'Unknown error';
    return new NextResponse(
      JSON.stringify({ error: 'Failed to process request', details }),
      { status, headers: { ...CORS_HEADERS, 'Content-Type': 'application/json' } }
    );
  }
}