import { NextResponse } from 'next/server';
import axios from 'axios';

const CORS_HEADERS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type, Authorization',
};

export async function OPTIONS() {
  return new NextResponse(null, { status: 204, headers: CORS_HEADERS });
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const {
      message,
      // AI Buddy fields (passed from client)
      buddy_id,
      buddy_name,
      // Model config – DO NOT CHANGE THE MODEL
      model = 'llama-3.3-70b-versatile',
      temperature = 0.7,
      max_tokens = 100,
    } = body || {};

    if (!message) {
      return new NextResponse(
        JSON.stringify({ error: 'message is required' }),
        { status: 400, headers: { ...CORS_HEADERS, 'Content-Type': 'application/json' } }
      );
    }

    // When a buddy is provided, the full prompt (including buddy persona instructions)
    // is already embedded in the `message` string by the client. We just forward it.
    // This keeps the motivation route simple and backward-compatible while allowing
    // buddy-flavoured responses when the client includes persona context in the message.
    const groqRes = await axios.post(
      'https://api.groq.com/openai/v1/chat/completions',
      {
        model,
        messages: [{ role: 'user', content: message }],
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
    return new NextResponse(
      JSON.stringify({ error: 'Failed to fetch motivation from LLM', details }),
      { status, headers: { ...CORS_HEADERS, 'Content-Type': 'application/json' } }
    );
  }
}