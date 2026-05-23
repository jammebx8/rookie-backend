// app/api/tts/route.ts

import { NextRequest, NextResponse } from 'next/server';
import Groq from 'groq-sdk';

export const runtime = 'nodejs';

// ─── Config ─────────────────────────────────────────────────────────────

const MODEL_ID = 'playai-tts';
const VOICE = 'Autumn';

const ALLOWED_ORIGIN =
  process.env.FRONTEND_URL ||
  'https://friday-kappa-ten.vercel.app';

// ─── Groq Client ───────────────────────────────────────────────────────

const groq = new Groq({
  apiKey: process.env.GROQ_API_KEY,
});

// ─── CORS ──────────────────────────────────────────────────────────────

function corsHeaders() {
  return {
    'Access-Control-Allow-Origin': ALLOWED_ORIGIN,
    'Access-Control-Allow-Methods': 'POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization',
  };
}

export async function OPTIONS() {
  return new NextResponse(null, {
    status: 204,
    headers: corsHeaders(),
  });
}

// ─── POST ──────────────────────────────────────────────────────────────

export async function POST(req: NextRequest) {
  try {
    // Validate API key
    if (!process.env.GROQ_API_KEY) {
      console.error('GROQ_API_KEY missing');

      return NextResponse.json(
        { error: 'Server misconfiguration' },
        {
          status: 500,
          headers: corsHeaders(),
        }
      );
    }

    // Parse request body
    const body = await req.json();
    const { text } = body;

    if (!text || typeof text !== 'string' || !text.trim()) {
      return NextResponse.json(
        { error: 'text is required' },
        {
          status: 400,
          headers: corsHeaders(),
        }
      );
    }

    // Safety limit
    const safeText = text.trim().slice(0, 5000);

    console.log('Generating TTS:', safeText.slice(0, 50));

    // Generate speech
    const speechResponse = await groq.audio.speech.create({
      model: MODEL_ID,
      voice: VOICE,
      response_format: 'wav',
      input: safeText,
    });

    // Convert response → buffer
    const arrayBuffer = await speechResponse.arrayBuffer();
    const audioBuffer = Buffer.from(arrayBuffer);

    console.log('TTS generated successfully');

    // Return audio
    return new NextResponse(audioBuffer, {
      status: 200,
      headers: {
        ...corsHeaders(),
        'Content-Type': 'audio/wav',
        'Content-Length': audioBuffer.length.toString(),
        'Cache-Control': 'no-cache',
      },
    });

  } catch (err: any) {
    console.error('FULL TTS ERROR:', err);
    console.error('ERROR MESSAGE:', err?.message);
    console.error('ERROR RESPONSE:', err?.response?.data);
  
    return NextResponse.json(
      {
        error: err?.message || 'TTS generation failed',
      },
      {
        status: 500,
        headers: corsHeaders(),
      }
    );
  }
  }
