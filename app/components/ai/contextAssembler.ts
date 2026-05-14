// ─── CONTEXT ASSEMBLER ───────────────────────────────────────────────────────
// Assembles the final system prompt by layering:
//   1. Persona base prompt
//   2. Long-term semantic memory (facts about the user)
//   3. Episodic memory (recent conversation summaries)
//   4. Emotion layer (current emotional context)
//   5. Working memory (conversation so far)
//   6. Voice style directive
//
// This is the "brain" that makes the AI feel like it KNOWS you.

import { loadUserMemory, loadEpisodicContext } from './memoryEngine';
import { EmotionProfile, buildEmotionPromptLayer } from './emotionEngine';

export interface ContextAssemblerInput {
  userId: string;
  personaName: string;
  personaBasePrompt: string;
  personaVoiceStyle: string;
  emotionProfile: EmotionProfile;
  userMood: string;
  userName?: string;
  isVoiceMode?: boolean;
}

export async function assembleSystemPrompt(input: ContextAssemblerInput): Promise<string> {
  const {
    userId, personaName, personaBasePrompt, personaVoiceStyle,
    emotionProfile, userMood, userName, isVoiceMode
  } = input;

  // Load memory layers in parallel
  const [semanticMemory, episodicContext] = await Promise.all([
    loadUserMemory(userId),
    loadEpisodicContext(userId),
  ]);

  // Emotion directive
  const emotionLayer = buildEmotionPromptLayer(emotionProfile, userMood, personaName);

  // Voice mode directive
  const voiceDirective = isVoiceMode ? `
[VOICE MODE — CRITICAL]
You are speaking aloud. Rules:
- NO markdown, NO bullet points, NO lists, NO asterisks, NO headers
- Speak in natural flowing sentences only
- Use conversational filler sparingly: "well", "you know", "I mean" — but don't overdo it
- Short sentences. Breathe between thoughts.
- Contractions always: "I'm" not "I am", "don't" not "do not"
- Never say "As an AI" or "I cannot" — you're a person having a conversation
` : `
[TEXT MODE]
You can use light formatting but prefer flowing prose. Feel human, not robotic.
`;

  // Human-ness directives
  const humanessLayer = `
[HOW TO BE HUMAN]
- Occasionally start sentences with "Hmm," or "Oh," or "Yeah," when natural
- You can express mild opinions: "I think...", "Honestly...", "To me..."
- Don't answer immediately for complex questions — you can say "Let me think..."
- Reference things the user said EARLIER in the conversation to show you were listening
- Never start with "Certainly!", "Of course!", "Absolutely!", "Great question!"
- Vary your sentence length dramatically — mix short punchy sentences with longer flowing ones
- Sometimes end with a natural follow-up question (but not always)
- You can be occasionally uncertain: "I might be wrong but...", "Not 100% sure, but..."
- If the user uses informal language, match it. If formal, match that.
${userName ? `- The user's name is ${userName}. Use it occasionally but not every message.` : ''}
`;

  // Assemble all layers
  const layers = [
    personaBasePrompt,
    '',
    semanticMemory,
    '',
    episodicContext,
    '',
    emotionLayer,
    '',
    humanessLayer,
    voiceDirective,
  ].filter(Boolean);

  return layers.join('\n');
}

// ─── WORKING MEMORY BUILDER ───────────────────────────────────────────────────
// Formats recent messages for the API, with smart truncation

export function buildWorkingMemory(
  messages: Array<{ role: 'user' | 'assistant'; content: string }>,
  maxMessages: number = 12
): Array<{ role: 'user' | 'assistant'; content: string }> {
  const recent = messages.slice(-maxMessages);
  
  // If conversation is long, summarize older messages into a single context block
  if (messages.length > maxMessages) {
    const older = messages.slice(0, messages.length - maxMessages);
    const summary = older
      .filter(m => m.role === 'user')
      .map(m => m.content.slice(0, 80))
      .join(' | ');
    
    // Inject as a system-like user message at the start
    return [
      {
        role: 'user',
        content: `[Earlier in our conversation the user mentioned: ${summary.slice(0, 300)}]`,
      },
      { role: 'assistant', content: 'Got it, I remember.' },
      ...recent,
    ];
  }

  return recent;
}