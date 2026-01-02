import { NextRequest } from 'next/server';

export const runtime = 'edge';

const CORS_HEADERS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type, Authorization',
};

export async function OPTIONS() {
  return new Response(null, { status: 204, headers: CORS_HEADERS });
}

export async function GET() {
  return new Response(
    JSON.stringify({ ok: true, message: 'Backend is working 🚀' }),
    { headers: { ...CORS_HEADERS, 'Content-Type': 'application/json' } }
  );
}

export async function POST(req: NextRequest) {
  const body = await req.json();
  return new Response(
    JSON.stringify({
      ok: true,
      received: body,
      reply: 'This will be AI output later 🤖',
    }),
    { headers: { ...CORS_HEADERS, 'Content-Type': 'application/json' } }
  );
}
