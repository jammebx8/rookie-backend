// app/api/transcribe/route.ts  (Next.js App Router)
// OR pages/api/transcribe.ts   (Next.js Pages Router — see bottom of file)
//
// Receives: multipart/form-data with field "audio" (Blob)
// Returns:  { text: string }

import { NextRequest, NextResponse } from 'next/server';
import Groq from 'groq-sdk';

// ─── CORS headers ─────────────────────────────────────────────────────────────
// Adjust ALLOWED_ORIGIN to your frontend domain in production.
const ALLOWED_ORIGIN = process.env.FRONTEND_URL || '*';

function corsHeaders() {
  return {
    'Access-Control-Allow-Origin': ALLOWED_ORIGIN,
    'Access-Control-Allow-Methods': 'POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization',
  };
}

// ─── OPTIONS (preflight) ──────────────────────────────────────────────────────
export async function OPTIONS() {
  return new NextResponse(null, { status: 204, headers: corsHeaders() });
}

// ─── POST ─────────────────────────────────────────────────────────────────────
export async function POST(req: NextRequest) {
  try {
    // Parse multipart form — Next.js App Router supports this natively
    const formData = await req.formData();
    const audioFile = formData.get('audio');

    if (!audioFile || !(audioFile instanceof Blob)) {
      return NextResponse.json(
        { error: 'Missing audio field in form data' },
        { status: 400, headers: corsHeaders() }
      );
    }

    // Convert the Blob → File so Groq SDK is happy
    const file = new File(
      [audioFile],
      'recording.webm',
      { type: audioFile.type || 'audio/webm' }
    );

    const groq = new Groq({
      apiKey: process.env.GROQ_API_KEY,
    });

    const transcription = await groq.audio.transcriptions.create({
      file,
      model: 'whisper-large-v3',
      temperature: 0,
      response_format: 'verbose_json',
    });

    return NextResponse.json(
      { text: transcription.text },
      { status: 200, headers: corsHeaders() }
    );

  } catch (err: any) {
    console.error('[/api/transcribe] error:', err);
    return NextResponse.json(
      { error: err.message || 'Transcription failed' },
      { status: 500, headers: corsHeaders() }
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// PAGES ROUTER ALTERNATIVE
// If you are using pages/api instead of app/, replace the entire file with:
// ─────────────────────────────────────────────────────────────────────────────
/*
import type { NextApiRequest, NextApiResponse } from 'next';
import Groq from 'groq-sdk';
import formidable, { IncomingForm } from 'formidable';
import fs from 'fs';

export const config = { api: { bodyParser: false } };

const ALLOWED_ORIGIN = process.env.FRONTEND_URL || '*';

function setCors(res: NextApiResponse) {
  res.setHeader('Access-Control-Allow-Origin', ALLOWED_ORIGIN);
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
}

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  setCors(res);
  if (req.method === 'OPTIONS') return res.status(204).end();
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });

  const form = new IncomingForm({ keepExtensions: true });
  form.parse(req, async (err, _fields, files) => {
    if (err) return res.status(400).json({ error: 'Form parse error' });

    const audioFile = Array.isArray(files.audio) ? files.audio[0] : files.audio;
    if (!audioFile) return res.status(400).json({ error: 'Missing audio field' });

    try {
      const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });
      const transcription = await groq.audio.transcriptions.create({
        file: fs.createReadStream(audioFile.filepath),
        model: 'whisper-large-v3',
        temperature: 0,
        response_format: 'verbose_json',
      });
      res.status(200).json({ text: transcription.text });
    } catch (e: any) {
      console.error('[/api/transcribe] error:', e);
      res.status(500).json({ error: e.message || 'Transcription failed' });
    }
  });
}
*/