import { NextRequest, NextResponse } from 'next/server';
import Groq from 'groq-sdk';
import { assembleSystemPrompt, buildWorkingMemory } from '../../components/ai/contextAssembler';
import { detectUserMood, computeAIEmotion, EmotionProfile } from '../../components/ai/emotionEngine';
import { extractAndStoreFacts, saveEpisodicSummary } from '../../components/ai/memoryEngine';

const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type, Authorization',
  'Access-Control-Max-Age': '86400',
};

export async function OPTIONS() {
  return new NextResponse(null, { status: 200, headers: corsHeaders });
}

export async function POST(request: NextRequest) {
  try {
    const {
      message,
      conversationId,
      personaId,
      personaName,
      personaBasePrompt,
      personaVoiceStyle,
      history,
      userId,
      userName,
      isVoiceMode,
      // Emotion state passed from client (persisted across turns)
      emotionHistory,
      currentEmotion,
    } = await request.json();

    if (!message || !personaBasePrompt) {
      return new NextResponse(JSON.stringify({ error: 'Missing required fields' }), {
        status: 400,
        headers: { 'Content-Type': 'application/json', ...corsHeaders },
      });
    }

    // ── 1. DETECT USER MOOD ────────────────────────────────────────────────
    const { mood: userMood, detectedEmotion, intensity } = detectUserMood(message);

    // ── 2. COMPUTE AI EMOTIONAL STATE ─────────────────────────────────────
    const emotionProfile: EmotionProfile = computeAIEmotion(
      detectedEmotion,
      personaId,
      emotionHistory || [],
      currentEmotion || 'neutral'
    );

    // ── 3. ASSEMBLE ENRICHED SYSTEM PROMPT ────────────────────────────────
    let systemPrompt = personaBasePrompt;
    if (userId) {
      systemPrompt = await assembleSystemPrompt({
        userId,
        personaName: personaName || 'Assistant',
        personaBasePrompt,
        personaVoiceStyle: personaVoiceStyle || 'warm',
        emotionProfile,
        userMood,
        userName,
        isVoiceMode: !!isVoiceMode,
      });
    } else {
      // No user ID — still inject emotion but skip memory
      const { buildEmotionPromptLayer } = await import('../../components/ai/emotionEngine');
      systemPrompt = personaBasePrompt + '\n\n' + buildEmotionPromptLayer(emotionProfile, userMood, personaName || 'Assistant');
    }

    // ── 4. BUILD WORKING MEMORY (smart truncation) ─────────────────────────
    const workingMemory = buildWorkingMemory(history || [], 14);

    const messages = [
      ...workingMemory,
      { role: 'user' as const, content: message },
    ];

    // ── 5. STREAM RESPONSE ─────────────────────────────────────────────────
    let fullResponse = '';

    const readable = new ReadableStream({
      async start(controller) {
        try {
          // Send emotion metadata first (client uses this to update its state)
          controller.enqueue(
            new TextEncoder().encode(
              `data: ${JSON.stringify({
                type: 'meta',
                emotionProfile,
                userMood,
              })}\n\n`
            )
          );

          const stream = await groq.chat.completions.create({
            model: 'llama-3.3-70b-versatile',
            messages: [{ role: 'system', content: systemPrompt }, ...messages],
            stream: true,
            temperature: 0.75,
            max_tokens: 1024,
            // Voice mode: shorter, punchier responses
            ...(isVoiceMode ? { max_tokens: 300 } : {}),
          });

          for await (const chunk of stream) {
            const content = chunk.choices[0]?.delta?.content || '';
            if (content) {
              fullResponse += content;
              controller.enqueue(
                new TextEncoder().encode(`data: ${JSON.stringify({ type: 'content', content })}\n\n`)
              );
            }
          }

          controller.enqueue(new TextEncoder().encode('data: [DONE]\n\n'));
          controller.close();

          // ── 6. POST-RESPONSE: MEMORY EXTRACTION (async, non-blocking) ──
          if (userId && fullResponse) {
            Promise.all([
              extractAndStoreFacts(userId, message, fullResponse),
              conversationId ? saveEpisodicSummary(
                conversationId,
                userId,
                [...(history || []), { role: 'user', content: message }, { role: 'assistant', content: fullResponse }],
                userMood
              ) : Promise.resolve(),
            ]).catch(err => console.error('Memory save error:', err));
          }

        } catch (error) {
          console.error('Stream error:', error);
          controller.error(error);
        }
      },
    });

    return new NextResponse(readable, {
      status: 200,
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        ...corsHeaders,
      },
    });

  } catch (error) {
    console.error('Chat API error:', error);
    return new NextResponse(
      JSON.stringify({ error: 'Failed to process chat request' }),
      { status: 500, headers: { 'Content-Type': 'application/json', ...corsHeaders } }
    );
  }
}