import { NextRequest, NextResponse } from 'next/server';
import Groq from 'groq-sdk';

const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });

const SYSTEM_PROMPT = `You are NutriLens AI, a food nutrition expert. Analyze the food provided and return ONLY valid JSON — no markdown, no preamble.

JSON structure (required, all fields):
{
  "food_name": "string",
  "description": "string",
  "serving_size": "string (e.g. '1 plate ~350g')",
  "confidence": "high|medium|low",
  "nutrition": {
    "calories": number,
    "protein_g": number,
    "carbs_g": number,
    "fat_g": number,
    "fiber_g": number,
    "sugar_g": number,
    "sodium_mg": number
  },
  "health_score": number (1-10),
  "advice": "string — 1-2 sentences, personalised to user goals and time of day",
  "warning": "string or null",
  "alternatives": ["string"] (2-3 items if health_score < 6, else [])
}`;

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { imageBase64, manualPrompt, userContext, scanMode = 'photo' } = body;

    if (!imageBase64 && !manualPrompt) {
      return NextResponse.json({ error: 'Provide imageBase64 or manualPrompt' }, { status: 400 });
    }

    const h = parseInt(userContext?.time_of_day?.split(':')[0] ?? '12');
    const timeLabel = h < 10 ? 'morning' : h < 14 ? 'midday' : h < 18 ? 'afternoon' : 'evening';

    const contextText = `
User profile:
- Diet: ${userContext?.diet_type ?? 'non_vegetarian'}
- Weight: ${userContext?.weight_kg ?? 70}kg
- Daily protein goal: ${userContext?.daily_protein_goal_g ?? 100}g (eaten so far: ${userContext?.todays_protein_so_far ?? 0}g)
- Daily calorie goal: ${userContext?.daily_calories_goal ?? 2000} kcal (eaten so far: ${userContext?.todays_calories_so_far ?? 0} kcal)
- Current time: ${timeLabel} (${userContext?.time_of_day ?? '12:00'})
- Meal: ${userContext?.meal_type ?? 'snack'}
- Scan mode: ${scanMode}
${userContext?.diet_type === 'vegetarian' || userContext?.diet_type === 'vegan' ? '- IMPORTANT: User does NOT eat meat/fish.' : ''}
${manualPrompt ? `\nUser says: "${manualPrompt}"` : ''}
`.trim();

    let messages: Groq.Chat.ChatCompletionMessageParam[];

    if (imageBase64) {
      // Vision model for images
      messages = [
        { role: 'system', content: SYSTEM_PROMPT },
        {
          role: 'user',
          content: [
            { type: 'text', text: contextText },
            { type: 'image_url', image_url: { url: `data:image/jpeg;base64,${imageBase64}` } },
          ] as any,
        },
      ];
    } else {
      // Text-only for manual input
      messages = [
        { role: 'system', content: SYSTEM_PROMPT },
        { role: 'user', content: contextText },
      ];
    }

    const model = imageBase64
      ? 'meta-llama/llama-4-scout-17b-16e-instruct'  // vision
      : 'llama-3.3-70b-versatile';                    // text

    const completion = await groq.chat.completions.create({
      model,
      messages,
      max_tokens: 900,
      temperature: 0.2,
    });

    const raw = completion.choices[0]?.message?.content ?? '';
    const cleaned = raw.replace(/```json|```/g, '').trim();

    let parsed;
    try {
      parsed = JSON.parse(cleaned);
    } catch {
      return NextResponse.json({ error: 'AI returned invalid JSON', raw }, { status: 422 });
    }

    return NextResponse.json({ success: true, data: parsed });
  } catch (err: any) {
    console.error('[analyze-food]', err?.message);
    return NextResponse.json({ error: err?.message ?? 'Server error' }, { status: 500 });
  }
}