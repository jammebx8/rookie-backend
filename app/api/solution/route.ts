import { NextRequest } from 'next/server';

export const runtime = 'edge';

const CORS_HEADERS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type, Authorization',
};

export async function OPTIONS() {
  return new Response(null, {
    status: 204,
    headers: CORS_HEADERS,
  });
}

export async function GET(req: NextRequest) {
  console.log('[motivation] GET invoked');
  return new Response(JSON.stringify({ ok: true, method: 'GET', route: '/api/motivation' }), {
    status: 200,
    headers: { ...CORS_HEADERS, 'Content-Type': 'application/json' },
  });
}

export async function POST(req: NextRequest) {
  try {
    console.log('[motivation] POST invoked');
    const body = await req.json().catch(() => null);
    return new Response(
      JSON.stringify({
        ok: true,
        method: 'POST',
        route: '/api/motivation',
        received: body,
      }),
      {
        status: 200,
        headers: { ...CORS_HEADERS, 'Content-Type': 'application/json' },
      }
    );
  } catch (err: any) {
    console.error('[motivation] POST error', err);
    return new Response(JSON.stringify({ error: 'server error', details: String(err) }), {
      status: 500,
      headers: { ...CORS_HEADERS, 'Content-Type': 'application/json' },
    });
  }
}