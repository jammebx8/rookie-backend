import { NextResponse } from 'next/server';
import axios from 'axios';

const CORS_HEADERS = {
  'Access-Control-Allow-Origin': '*', // change to your allowed origin in production
  'Access-Control-Allow-Methods': 'POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type, Authorization',
};

export async function OPTIONS() {
  return new NextResponse(null, {
    status: 204,
    headers: CORS_HEADERS,
  });
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { 
      message, 
      characterId,
      isCorrect,
      model = 'llama-3.3-70b-versatile', 
      temperature = 0.7, 
      max_tokens = 50 
    } = body || {};

    if (!message) {
      return new NextResponse(JSON.stringify({ error: 'message is required' }), {
        status: 400,
        headers: { ...CORS_HEADERS, 'Content-Type': 'application/json' },
      });
    }

    // Enhanced prompt based on character and result
    let enhancedPrompt = message;

    // Optional: Add context based on whether answer was correct/incorrect
    if (typeof isCorrect === 'boolean') {
      if (isCorrect) {
        enhancedPrompt += ' The student got the answer CORRECT. Be encouraging and motivating.';
      } else {
        enhancedPrompt += ' The student got the answer WRONG. Be supportive and encouraging, help them stay motivated.';
      }
    }

    const groqRes = await axios.post(
      'https://api.groq.com/openai/v1/chat/completions',
      {
        model,
        messages: [{ role: 'user', content: enhancedPrompt }],
        temperature,
        max_tokens,
      },
      {
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${process.env.GROQ_API_KEY}`,
        },
        timeout: 120000,
      }
    );
 
    return new NextResponse(JSON.stringify(groqRes.data), {
      status: 200,
      headers: { ...CORS_HEADERS, 'Content-Type': 'application/json' },
    });
  } catch (err: any) {
    console.error('motivation route error:', err?.response?.data || err?.message || err);
    const status = err?.response?.status || 500;
    const details = err?.response?.data || err?.message || 'Unknown error';
    return new NextResponse(JSON.stringify({ error: 'Failed to fetch motivation from LLM', details }), {
      status,
      headers: { ...CORS_HEADERS, 'Content-Type': 'application/json' },
    });
  }
}