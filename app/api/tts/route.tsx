// app/api/tts/route.ts  (Next.js App Router)
//
// Receives: JSON { text: string }
// Returns:  audio/mpeg stream (mp3)

import { NextRequest, NextResponse } from 'next/server';
import { ElevenLabsClient } from '@elevenlabs/elevenlabs-js';

// ─── Config ───────────────────────────────────────────────────────────────────
// Set ELEVENLABS_VOICE_ID in .env to override the default.
// Default: "JBFqnCBsd6RMkjVDRZzb" (George — a clear, neutral English voice)
const VOICE_ID = '21m00Tcm4TlvDq8ikWAM';
const MODEL_ID = 'eleven_multilingual_v2';
const OUTPUT_FORMAT = 'mp3_44100_128';



// ─── CORS helpers ─────────────────────────────────────────────────────────────
const ALLOWED_ORIGIN = process.env.FRONTEND_URL || '*';

function corsHeaders() {
  return {
    'Access-Control-Allow-Origin': ALLOWED_ORIGIN,
    'Access-Control-Allow-Methods': 'POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization',
  };
}

export async function OPTIONS() {
  return new NextResponse(null, { status: 204, headers: corsHeaders() });
}

// ─── POST ─────────────────────────────────────────────────────────────────────
export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { text } = body;

    if (!text?.trim()) {
      return NextResponse.json(
        { error: 'text is required' },
        { status: 400, headers: corsHeaders() }
      );
    }

    // Trim to 5000 chars — ElevenLabs limit per request
    const safeText = text.trim().slice(0, 5000);

    const elevenlabs = new ElevenLabsClient({
      apiKey: process.env.ELEVENLABS_API_KEY,
    });

    // Returns a Node.js Readable / Web ReadableStream depending on env
    const audioStream = await elevenlabs.textToSpeech.convert(VOICE_ID, {
      text: safeText,
      modelId: MODEL_ID,
      outputFormat: OUTPUT_FORMAT,
    });

    // ElevenLabs SDK returns a Web ReadableStream<Uint8Array> in edge/Node18+
    // We pipe it straight through to the response.
    // If it comes back as a Node Readable, convert it first.
    let webStream: ReadableStream<Uint8Array>;

    if (audioStream instanceof ReadableStream) {
      webStream = audioStream as ReadableStream<Uint8Array>;
    } else {
      // Node.js Readable → Web ReadableStream
      webStream = new ReadableStream<Uint8Array>({
        start(controller) {
          (audioStream as any).on('data', (chunk: Buffer) => {
            controller.enqueue(new Uint8Array(chunk));
          });
          (audioStream as any).on('end', () => controller.close());
          (audioStream as any).on('error', (err: Error) => controller.error(err));
        },
      });
    }

    return new NextResponse(webStream, {
      status: 200,
      headers: {
        ...corsHeaders(),
        'Content-Type': 'audio/mpeg',
        'Cache-Control': 'no-cache',
        'Transfer-Encoding': 'chunked',
      },
    });

  } catch (err: any) {
    console.error('[/api/tts] error:', err);
    return NextResponse.json(
      { error: err.message || 'TTS failed' },
      { status: 500, headers: corsHeaders() }
    );
  }
}