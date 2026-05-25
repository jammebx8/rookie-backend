// app/api/chat/route.ts  (Next.js App Router)
//
// Receives: JSON { message, conversationId, personaId, personaName,
//                  personaSystemPrompt, history, userName }
// Returns:  SSE stream of { type: 'content', content: string } chunks
//           terminated with "data: [DONE]\n\n"

import { NextRequest, NextResponse } from 'next/server';

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
    const systemPrompt = `${userContext}${personaSystemPrompt || `You are ${personaName || 'an AI assistant'}. Be helpful and engaging.`}`;

    // Build message array
    const messages: Array<{ role: 'system' | 'user' | 'assistant'; content: string }> = [
      { role: 'system', content: systemPrompt },
      ...history.map((m: { role: string; content: string }) => ({
        role: m.role as 'user' | 'assistant',
        content: m.content,
      })),
      { role: 'user', content: message.trim() },
    ];

    // ─── OpenRouter fetch (streaming) ─────────────────────────────────────────
    const openRouterRes = await fetch('https://openrouter.ai/api/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${process.env.OPENROUTER_API_KEY}`,
        'Content-Type': 'application/json',
        'HTTP-Referer': process.env.FRONTEND_URL || 'http://localhost:3000',
        'X-Title': personaName || 'AI Companion',
      },
      body: JSON.stringify({
        model: 'cognitivecomputations/dolphin-mistral-24b-venice-edition:free',
        messages,
        temperature: 0.9,
        max_tokens: 1024,
        top_p: 1,
        stream: true,
      }),
    });

    if (!openRouterRes.ok) {
      const errText = await openRouterRes.text();
      console.error('[/api/chat] OpenRouter error:', errText);
      return NextResponse.json(
        { error: `OpenRouter error: ${openRouterRes.status}` },
        { status: 500, headers: corsHeaders() }
      );
    }

    // ─── Pipe SSE from OpenRouter → client ────────────────────────────────────
    const encoder = new TextEncoder();
    const decoder = new TextDecoder();

    const readableStream = new ReadableStream({
      async start(controller) {
        const reader = openRouterRes.body!.getReader();
        let buffer = '';

        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            // Keep last (possibly incomplete) line in buffer
            buffer = lines.pop() ?? '';

            for (const line of lines) {
              const trimmed = line.trim();
              if (!trimmed || trimmed === 'data: [DONE]') {
                if (trimmed === 'data: [DONE]') {
                  controller.enqueue(encoder.encode('data: [DONE]\n\n'));
                }
                continue;
              }

              if (trimmed.startsWith('data: ')) {
                try {
                  const json = JSON.parse(trimmed.slice(6));
                  const content = json.choices?.[0]?.delta?.content;
                  if (content) {
                    const sseChunk = `data: ${JSON.stringify({ type: 'content', content })}\n\n`;
                    controller.enqueue(encoder.encode(sseChunk));
                  }
                } catch {
                  // skip malformed chunks
                }
              }
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
          reader.releaseLock();
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