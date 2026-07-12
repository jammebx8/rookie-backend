// app/api/tts/route.ts

import { NextRequest, NextResponse } from 'next/server';

export const runtime = 'nodejs';

// "Bella" — expressive, natural female voice. Swap for any ElevenLabs voice_id you like.
const VOICE_ID = process.env.ELEVENLABS_VOICE_ID || 'ftDdhfYtmfGP0tFlBYA1';

// eleven_multilingual_v2 gives the most natural, emotionally expressive delivery.
// If you have access to eleven_v3 (alpha), that one is even more emotive.
const MODEL_ID = process.env.ELEVENLABS_MODEL_ID || 'eleven_multilingual_v2';

const ALLOWED_ORIGIN =
  process.env.FRONTEND_URL ||
  'https://friday-kappa-ten.vercel.app';

const ELEVENLABS_TTS_URL = `https://api.elevenlabs.io/v1/text-to-speech/${VOICE_ID}`;

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

    // Safety limit
    const safeText = text.trim().slice(0, 5000);

    console.log('Generating TTS:', safeText.slice(0, 50));

    // Generate speech via ElevenLabs
    const speechResponse = await fetch(
      `${ELEVENLABS_TTS_URL}?output_format=mp3_44100_128`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'xi-api-key': process.env.ELEVENLABS_API_KEY,
          Accept: 'audio/mpeg',
        },
        body: JSON.stringify({
          text: safeText,
          model_id: MODEL_ID,
          voice_settings: {
            // Lower stability + higher style = more emotional, expressive range.
            // Push stability up a bit if the voice starts sounding unstable/glitchy.
            stability: 0.45,
            similarity_boost: 0.85,
            style: 0.65,
            use_speaker_boost: true,
          },
        }),
      }
    );

    if (!speechResponse.ok) {
      const errText = await speechResponse.text().catch(() => '');
      console.error('ElevenLabs error:', speechResponse.status, errText);

      return NextResponse.json(
        { error: `TTS provider error (${speechResponse.status})` },
        {
          status: 502,
          headers: corsHeaders(),
        }
      );
    }

    // Convert response → buffer
    const arrayBuffer = await speechResponse.arrayBuffer();
    const audioBuffer = Buffer.from(arrayBuffer);

    console.log('TTS generated successfully');

    // Return audio
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
    console.error('FULL TTS ERROR:', err);
    console.error('ERROR MESSAGE:', err?.message);

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