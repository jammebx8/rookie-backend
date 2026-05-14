// ─── MEMORY ENGINE ────────────────────────────────────────────────────────────
// Manages 3-layer memory: working (in-session), episodic (per conversation),
// and semantic (long-term user facts extracted across all sessions)

import { supabase } from '@/public/src/utils/supabase';

export interface MemoryFact {
  id?: string;
  user_id: string;
  category: 'preference' | 'fact' | 'goal' | 'emotion' | 'routine' | 'relationship';
  key: string;       // e.g. "favorite_subject", "stress_about"
  value: string;     // e.g. "mathematics", "JEE mains in 3 months"
  confidence: number; // 0–1, how sure we are
  last_seen: string;  // ISO
  times_reinforced: number;
}

export interface EpisodicSummary {
  conversation_id: string;
  user_id: string;
  summary: string;        // 2–3 sentence recap
  mood: string;           // detected mood during conversation
  topics: string[];       // main topics discussed
  created_at: string;
}

export interface WorkingMemory {
  recentMessages: Array<{ role: 'user' | 'assistant'; content: string }>;
  detectedMood: string;
  topicsThisSession: string[];
  emotionHistory: string[]; // last 5 emotions detected
}

// ─── EXTRACT FACTS FROM LLM RESPONSE ─────────────────────────────────────────

export async function extractAndStoreFacts(
  userId: string,
  userMessage: string,
  assistantResponse: string
): Promise<void> {
  // Simple regex-based extraction for free (no extra API call)
  const facts: Partial<MemoryFact>[] = [];

  // Preference patterns
  const preferPatterns = [
    { re: /i (?:love|really like|enjoy|prefer) (.+?)(?:\.|,|$)/gi, cat: 'preference' as const },
    { re: /(?:i'm|i am) (?:good at|interested in|passionate about) (.+?)(?:\.|,|$)/gi, cat: 'preference' as const },
    { re: /my (?:favorite|fav) (?:\w+ is) (.+?)(?:\.|,|$)/gi, cat: 'preference' as const },
  ];

  // Goal patterns
  const goalPatterns = [
    { re: /(?:i want to|i'm trying to|my goal is to|i need to) (.+?)(?:\.|,|$)/gi, cat: 'goal' as const },
    { re: /(?:i'm preparing for|studying for|targeting) (.+?)(?:\.|,|$)/gi, cat: 'goal' as const },
  ];

  // Fact patterns
  const factPatterns = [
    { re: /(?:i'm|i am) (?:a |an )?(\w[\w\s]+?) (?:student|from|in)/gi, cat: 'fact' as const },
    { re: /(?:my name is|call me) ([A-Z][a-z]+)/g, cat: 'fact' as const },
    { re: /i (?:study|go to|attend) (.+?)(?:\.|,|$)/gi, cat: 'fact' as const },
  ];

  const allPatterns = [...preferPatterns, ...goalPatterns, ...factPatterns];

  for (const { re, cat } of allPatterns) {
    let match;
    while ((match = re.exec(userMessage)) !== null) {
      const value = match[1]?.trim();
      if (value && value.length > 2 && value.length < 100) {
        facts.push({
          user_id: userId,
          category: cat,
          key: cat + '_' + value.toLowerCase().replace(/\s+/g, '_').slice(0, 30),
          value,
          confidence: 0.7,
          last_seen: new Date().toISOString(),
          times_reinforced: 1,
        });
      }
    }
  }

  if (facts.length === 0) return;

  // Upsert facts (update if key exists, increment reinforcement)
  for (const fact of facts) {
    try {
      const { data: existing } = await supabase
        .from('ai_memory_facts')
        .select('id, times_reinforced, confidence')
        .eq('user_id', userId)
        .eq('key', fact.key)
        .single();

      if (existing) {
        await supabase
          .from('ai_memory_facts')
          .update({
            value: fact.value,
            confidence: Math.min(1, existing.confidence + 0.1),
            times_reinforced: existing.times_reinforced + 1,
            last_seen: fact.last_seen,
          })
          .eq('id', existing.id);
      } else {
        await supabase.from('ai_memory_facts').insert(fact);
      }
    } catch (_) {
      // Silently fail — memory is enhancement, not critical path
    }
  }
}

// ─── LOAD USER MEMORY FOR PROMPT INJECTION ────────────────────────────────────

export async function loadUserMemory(userId: string): Promise<string> {
  try {
    const { data: facts } = await supabase
      .from('ai_memory_facts')
      .select('category, key, value, confidence')
      .eq('user_id', userId)
      .gte('confidence', 0.5)
      .order('times_reinforced', { ascending: false })
      .limit(20);

    if (!facts || facts.length === 0) return '';

    const grouped: Record<string, string[]> = {};
    for (const f of facts) {
      if (!grouped[f.category]) grouped[f.category] = [];
      grouped[f.category].push(f.value);
    }

    const lines: string[] = ['[WHAT YOU KNOW ABOUT THIS USER]'];
    for (const [cat, values] of Object.entries(grouped)) {
      lines.push(`${cat.toUpperCase()}: ${values.join(', ')}`);
    }
    lines.push('[USE THIS TO PERSONALIZE — DO NOT MENTION YOU HAVE A MEMORY SYSTEM]');

    return lines.join('\n');
  } catch {
    return '';
  }
}

// ─── SAVE EPISODIC SUMMARY ────────────────────────────────────────────────────

export async function saveEpisodicSummary(
  conversationId: string,
  userId: string,
  messages: Array<{ role: string; content: string }>,
  detectedMood: string
): Promise<void> {
  if (messages.length < 4) return; // Not worth summarizing tiny convos

  // Build a compact summary from last messages
  const recentExchange = messages.slice(-6).map(m =>
    `${m.role === 'user' ? 'User' : 'AI'}: ${m.content.slice(0, 120)}`
  ).join('\n');

  const topicWords = messages
    .filter(m => m.role === 'user')
    .map(m => m.content)
    .join(' ')
    .toLowerCase()
    .split(/\W+/)
    .filter(w => w.length > 4)
    .slice(0, 10);

  try {
    await supabase.from('ai_episodic_memory').upsert({
      conversation_id: conversationId,
      user_id: userId,
      summary: recentExchange.slice(0, 500),
      mood: detectedMood,
      topics: topicWords,
      created_at: new Date().toISOString(),
    }, { onConflict: 'conversation_id' });
  } catch (_) {}
}

// ─── LOAD EPISODIC CONTEXT (last 3 conversations) ────────────────────────────

export async function loadEpisodicContext(userId: string): Promise<string> {
  try {
    const { data } = await supabase
      .from('ai_episodic_memory')
      .select('summary, mood, topics, created_at')
      .eq('user_id', userId)
      .order('created_at', { ascending: false })
      .limit(3);

    if (!data || data.length === 0) return '';

    const lines = ['[RECENT CONVERSATION HISTORY]'];
    for (const ep of data) {
      const ago = getRelativeTime(ep.created_at);
      lines.push(`${ago}: User was ${ep.mood}. Topics: ${ep.topics?.slice(0, 4).join(', ')}.`);
    }
    return lines.join('\n');
  } catch {
    return '';
  }
}

function getRelativeTime(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  const hours = Math.floor(diff / 3600000);
  if (hours < 2) return 'Recently';
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}