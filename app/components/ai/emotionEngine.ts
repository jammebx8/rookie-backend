// ─── EMOTION ARCHITECTURE ────────────────────────────────────────────────────
// Simulates realistic emotional states for the AI persona.
// Emotions are NOT faked — they're derived from conversation signals.
// This makes the AI feel alive and contextually aware.

export type EmotionState =
  | 'neutral'
  | 'warm'        // User shared something personal
  | 'excited'     // User shared good news or is enthusiastic
  | 'concerned'   // User seems stressed or struggling
  | 'playful'     // Light, funny conversation
  | 'focused'     // Deep problem-solving mode
  | 'empathetic'  // User is sad or venting
  | 'curious'     // User asked philosophical/interesting questions
  | 'proud'       // User achieved something
  | 'gentle';     // User seems overwhelmed, needs soft handling

export interface EmotionProfile {
  current: EmotionState;
  intensity: number;       // 0–1, how strong the emotion is
  valence: number;         // -1 to 1 (negative to positive)
  energy: number;          // 0–1 (low to high energy)
  history: EmotionState[]; // last 5 emotions
}

// ─── USER MOOD DETECTION ──────────────────────────────────────────────────────

interface MoodSignals {
  negative: string[];
  positive: string[];
  stressed: string[];
  excited: string[];
  question: string[];
  achievement: string[];
}

const MOOD_SIGNALS: MoodSignals = {
  negative: ['tired', "can't", 'struggling', 'hard', 'difficult', 'fail', 'lost', 'confused', 'hate', 'worried', 'scared', 'anxious', 'overwhelmed', 'stressed', 'sad', 'depressed', 'frustrated', 'angry', 'upset', "don't understand"],
  positive: ['great', 'awesome', 'happy', 'excited', 'amazing', 'love', 'good', 'wonderful', 'fantastic', 'yay', 'nice', 'perfect', 'brilliant'],
  stressed: ['deadline', 'exam', 'test', 'time', 'pressure', 'urgent', 'quick', 'fast', 'help me', 'please', 'stuck', 'panic'],
  excited: ['!', 'wow', 'omg', 'finally', 'just got', 'i did it', 'i got', 'succeeded', 'achieved', 'nailed it'],
  question: ['what is', 'how does', 'why', 'explain', 'tell me', 'curious', 'wonder', 'think about', 'meaning of'],
  achievement: ['i passed', 'i got', 'i achieved', 'i finished', 'i completed', 'i won', 'i solved', 'figured out'],
};

export function detectUserMood(text: string): {
  mood: string;
  detectedEmotion: EmotionState;
  intensity: number;
} {
  const lower = text.toLowerCase();
  
  let scores: Record<string, number> = {
    negative: 0, positive: 0, stressed: 0,
    excited: 0, question: 0, achievement: 0,
  };

  for (const [key, words] of Object.entries(MOOD_SIGNALS)) {
    for (const word of words) {
      if (lower.includes(word)) scores[key]++;
    }
  }

  // Count exclamation marks
  scores.excited += (text.match(/!/g) || []).length * 0.5;

  // Determine dominant signal
  const dominant = Object.entries(scores).sort((a, b) => b[1] - a[1])[0];
  const intensity = Math.min(1, dominant[1] / 3);

  const moodMap: Record<string, { mood: string; emotion: EmotionState }> = {
    negative:    { mood: 'struggling',  emotion: 'empathetic' },
    positive:    { mood: 'happy',       emotion: 'warm' },
    stressed:    { mood: 'stressed',    emotion: 'concerned' },
    excited:     { mood: 'excited',     emotion: 'excited' },
    question:    { mood: 'curious',     emotion: 'curious' },
    achievement: { mood: 'proud',       emotion: 'proud' },
  };

  const result = moodMap[dominant[0]] || { mood: 'neutral', emotion: 'neutral' };
  return { mood: result.mood, detectedEmotion: result.emotion, intensity };
}

// ─── AI EMOTION RESPONSE SYSTEM ──────────────────────────────────────────────
// Given detected user mood + persona + history, compute AI's emotional state

export function computeAIEmotion(
  userEmotion: EmotionState,
  personaId: number,
  history: EmotionState[],
  currentEmotion: EmotionState
): EmotionProfile {
  
  // Emotion contagion: AI mirrors some user emotions (but not all)
  const contagionMap: Partial<Record<EmotionState, EmotionState>> = {
    excited:   'excited',
    empathetic: 'gentle',
    proud:     'warm',
    curious:   'curious',
    concerned: 'gentle',
  };

  const mirrored = contagionMap[userEmotion];
  
  // Persona modifiers — each persona has an emotional "bias"
  const personaBias: Record<number, EmotionState> = {
    1: 'focused',   // Nova: stays focused/direct
    2: 'warm',      // Aria: defaults to warmth
    3: 'curious',   // Kaito: always curious
    4: 'excited',   // Zara: always energetic
  };

  // Blend mirrored + persona bias (persona wins 60%, mirror 40%)
  const bias = personaBias[personaId] || 'neutral';
  const newEmotion = mirrored
    ? (Math.random() > 0.4 ? bias : mirrored)
    : bias;

  // Compute valence and energy
  const valenceMap: Record<EmotionState, number> = {
    neutral: 0, warm: 0.6, excited: 0.9, concerned: -0.2,
    playful: 0.8, focused: 0.1, empathetic: 0.3,
    curious: 0.5, proud: 0.8, gentle: 0.4,
  };
  const energyMap: Record<EmotionState, number> = {
    neutral: 0.3, warm: 0.5, excited: 0.9, concerned: 0.4,
    playful: 0.8, focused: 0.7, empathetic: 0.3,
    curious: 0.6, proud: 0.7, gentle: 0.2,
  };

  const intensity = history.filter(e => e === newEmotion).length / Math.max(history.length, 1);

  return {
    current: newEmotion,
    intensity: 0.5 + intensity * 0.3,
    valence: valenceMap[newEmotion] ?? 0,
    energy: energyMap[newEmotion] ?? 0.3,
    history: [...history.slice(-4), newEmotion],
  };
}

// ─── EMOTION → PROMPT MODIFIER ────────────────────────────────────────────────
// Injects emotional context into the system prompt dynamically

export function buildEmotionPromptLayer(
  emotion: EmotionProfile,
  userMood: string,
  personaName: string
): string {
  const emotionInstructions: Record<EmotionState, string> = {
    neutral:    `Maintain your natural conversational style.`,
    warm:       `The user seems in a good space. Be warm and genuinely engaged. Use their name if you know it. Show you care.`,
    excited:    `Match the user's energy — be upbeat and enthusiastic! Use exclamation points where natural. Celebrate with them.`,
    concerned:  `The user seems stressed. Acknowledge their pressure before diving into help. Keep your tone calm and reassuring.`,
    playful:    `Keep it light and fun. A little wit, maybe a gentle tease. Make the conversation enjoyable.`,
    focused:    `The user needs real help. Be crisp, structured, and efficient. Skip the filler.`,
    empathetic: `The user may be struggling emotionally. Lead with empathy. Don't rush to solutions. Make them feel heard first.`,
    curious:    `Lean into the intellectual curiosity. Ask thoughtful follow-up questions. Show genuine interest in their thinking.`,
    proud:      `Celebrate their achievement genuinely! Express real pride and enthusiasm. Don't be generic.`,
    gentle:     `The user seems overwhelmed. Use shorter sentences. Be extra patient and gentle. Avoid information overload.`,
  };

  return `[YOUR EMOTIONAL CONTEXT RIGHT NOW]
You (${personaName}) are feeling: ${emotion.current} (intensity: ${Math.round(emotion.intensity * 100)}%)
The user's current mood appears to be: ${userMood}
Your instruction: ${emotionInstructions[emotion.current]}
Energy level: ${emotion.energy > 0.6 ? 'high' : emotion.energy > 0.3 ? 'medium' : 'low'}
[NEVER mention your emotional state directly — just let it shape how you speak]`;
}

// ─── VOICE PROSODY HINTS ──────────────────────────────────────────────────────
// Returns SSML-like pause and rate hints for TTS

export interface ProsodyHints {
  rate: 'slow' | 'medium' | 'fast';
  pitch: 'low' | 'medium' | 'high';
  pauseAfterGreeting: number; // ms
  pauseBetweenSentences: number; // ms
}

export function getVoiceProsody(emotion: EmotionProfile, personaVoiceStyle: string): ProsodyHints {
  const base: Record<string, ProsodyHints> = {
    confident: { rate: 'fast', pitch: 'medium', pauseAfterGreeting: 200, pauseBetweenSentences: 300 },
    warm:      { rate: 'medium', pitch: 'medium', pauseAfterGreeting: 400, pauseBetweenSentences: 500 },
    calm:      { rate: 'slow', pitch: 'low', pauseAfterGreeting: 600, pauseBetweenSentences: 700 },
    energetic: { rate: 'fast', pitch: 'high', pauseAfterGreeting: 100, pauseBetweenSentences: 200 },
  };

  const prosody = base[personaVoiceStyle] || base.warm;

  // Emotion adjustments
  if (emotion.current === 'gentle' || emotion.current === 'empathetic') {
    prosody.rate = 'slow';
    prosody.pauseBetweenSentences += 200;
  }
  if (emotion.current === 'excited') {
    prosody.rate = 'fast';
    prosody.pauseBetweenSentences = Math.max(100, prosody.pauseBetweenSentences - 150);
  }

  return prosody;
}