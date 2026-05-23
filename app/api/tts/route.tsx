// app/api/tts/route.ts

import { NextRequest, NextResponse } from 'next/server';
import { ElevenLabsClient } from '@elevenlabs/elevenlabs-js';

export const runtime = 'nodejs';

// ─── Config ─────────────────────────────────────────────────────────────

const VOICE_ID = '21m00Tcm4TlvDq8ikWAM';
const MODEL_ID = 'eleven_multilingual_v2';
const OUTPUT_FORMAT = 'mp3_44100_128';

const ALLOWED_ORIGIN =
  process.env.FRONTEND_URL ||
  'https://friday-kappa-ten.vercel.app';

// ─── CORS ───────────────────────────────────────────────────────────────

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

// ─── POST ───────────────────────────────────────────────────────────────

export async function POST(req: NextRequest) {
  try {
    // Validate API key
    if (!process.env.ELEVENLABS_API_KEY) {
      console.error('ELEVENLABS_API_KEY missing');

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

    // ElevenLabs limit safety
    const safeText = text.trim().slice(0, 5000);

    // Create client
    const elevenlabs = new ElevenLabsClient({
      apiKey: process.env.ELEVENLABS_API_KEY,
    });

    console.log('Generating TTS:', safeText.slice(0, 50));

    // Generate audio
    const audioStream = await elevenlabs.textToSpeech.convert(
      VOICE_ID,
      {
        text: safeText,
        modelId: MODEL_ID,
        outputFormat: OUTPUT_FORMAT,
      }
    );

    // Convert async iterable stream → buffer
    const chunks: Buffer[] = [];

    for await (const chunk of audioStream as any) {
      chunks.push(Buffer.from(chunk));
    }

    const audioBuffer = Buffer.concat(chunks);

    console.log('TTS generated successfully');

    // Return mp3
    return new NextResponse(audioBuffer, {
      status: 200,
      headers: {
        ...corsHeaders(),
        'Content-Type': 'audio/mpeg',
        'Content-Length': audioBuffer.length.toString(),
        'Cache-Control': 'no-cache',
      },
    });

  } catch (err: any) {
    console.error('[/api/tts] ERROR:', err);

    return NextResponse.json(
      {
        error:
          err?.message ||
          'TTS generation failed',
      },
      {
        status: 500,
        headers: corsHeaders(),
      }
    );
  }
}