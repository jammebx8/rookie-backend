// app/api/chat/route.ts  (Next.js App Router)
//
// Receives: JSON { message, conversationId, personaId, personaName,
//                  personaSystemPrompt, history, userName }
// Returns:  SSE stream of { type: 'content', content: string } chunks
//           terminated with "data: [DONE]\n\n"

import { NextRequest, NextResponse } from 'next/server';
import { OpenRouter } from '@openrouter/sdk';

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

// ─── OpenRouter client ────────────────────────────────────────────────────────
const openrouter = new OpenRouter({
  apiKey: process.env.OPENROUTER_API_KEY!,
});

// ─── POST ─────────────────────────────────────────────────────────────────────
export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const {
      message,
      personaName,
      personaSystemPrompt,
      history = [],
      userName,
    } = body;

    if (!message?.trim()) {
      return NextResponse.json(
        { error: 'message is required' },
        { status: 400, headers: corsHeaders() }
      );
    }

    // Build the system prompt
    const userContext = userName ? `The user's name is ${userName}. ` : '';
    const systemPrompt = `${userContext}${
      personaSystemPrompt ||
      `You are ${personaName || 'an AI assistant'}. Be helpful and engaging.`
    }`;

    // Build message array
    const messages: Array<{ role: 'system' | 'user' | 'assistant'; content: string }> = [
      { role: 'system', content: systemPrompt },
      ...history.map((m: { role: string; content: string }) => ({
        role: m.role as 'user' | 'assistant',
        content: m.content,
      })),
      { role: 'user', content: message.trim() },
    ];

    // ─── Stream from OpenRouter SDK ───────────────────────────────────────────
    const stream = await openrouter.chat.send({
      model: 'cognitivecomputations/dolphin-mistral-24b-venice-edition:free',
      messages,
      temperature: 0.9,
      max_tokens: 1024,
      top_p: 1,
      stream: true,
    });

    // ─── Pipe SDK stream → SSE response ───────────────────────────────────────
    const encoder = new TextEncoder();

    const readableStream = new ReadableStream({
      async start(controller) {
        try {
          for await (const chunk of stream) {
            const content = chunk.choices[0]?.delta?.content;
            if (content) {
              const sseChunk = `data: ${JSON.stringify({ type: 'content', content })}\n\n`;
              controller.enqueue(encoder.encode(sseChunk));
            }
          }
          controller.enqueue(encoder.encode('data: [DONE]\n\n'));
        } catch (err) {
          console.error('[/api/chat] stream error:', err);
          controller.enqueue(
            encoder.encode(
              `data: ${JSON.stringify({ type: 'error', error: 'Stream interrupted' })}\n\n`
            )
          );
        } finally {
          controller.close();
        }
      },
    });

    return new NextResponse(readableStream, {
      status: 200,
      headers: {
        ...corsHeaders(),
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache, no-transform',
        'X-Accel-Buffering': 'no',
        Connection: 'keep-alive',
      },
    });

  } catch (err: any) {
    console.error('[/api/chat] error:', err);
    return NextResponse.json(
      { error: err.message || 'Chat failed' },
      { status: 500, headers: corsHeaders() }
    );
  }
}