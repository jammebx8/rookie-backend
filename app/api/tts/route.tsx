import { NextRequest, NextResponse } from 'next/server';

interface TTSRequest {
  text: string;
  voiceStyle?: string;
}

export async function POST(request: NextRequest) {
  try {
    const body: TTSRequest = await request.json();

    if (!body.text) {
      return NextResponse.json(
        { error: 'Text is required' },
        { status: 400 }
      );
    }

    if (!process.env.CANOPY_API_KEY) {
      return NextResponse.json(
        { error: 'CANOPY_API_KEY not configured' },
        { status: 500 }
      );
    }

    // Map voice styles to Canopy Labs voice IDs
    const voiceMap: Record<string, string> = {
      confident: '1', // Nova - confident voice
      warm: '2', // Aria - warm voice
      calm: '3', // Kaito - calm voice
      energetic: '4', // Zara - energetic voice
    };

    const voiceId = voiceMap[body.voiceStyle || 'confident'] || '1';

    // Call Canopy Labs Orpheus API
    const response = await fetch('https://api.canopylabs.ai/v1/audio/speech', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${process.env.CANOPY_API_KEY}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        text: body.text,
        voice_id: voiceId,
        model: 'orpheus-v1-english',
        speed: 1.0,
        language: 'en',
      }),
    });

    if (!response.ok) {
      const error = await response.text();
      console.error('Canopy API error:', error);
      return NextResponse.json(
        { error: 'Failed to generate speech' },
        { status: response.status }
      );
    }

    // Get audio data
    const audioBuffer = await response.arrayBuffer();

    return new Response(audioBuffer, {
      headers: {
        'Content-Type': 'audio/mpeg',
        'Cache-Control': 'no-cache',
      },
    });
  } catch (error) {
    console.error('TTS API error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    );
  }
}
