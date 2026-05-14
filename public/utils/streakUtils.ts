// lib/streakUtils.ts
import { supabase } from '../utils/supabase'; // adjust path

export const STREAK_KEY = 'rookie_streak_data';

export type StreakData = {
  current_streak: number;
  longest_streak: number;
  last_attempt_date: string | null;
  streak_days: string[]; // ISO date strings "YYYY-M-D"
};

function toDateKey(date: Date = new Date()) {
  return `${date.getFullYear()}-${date.getMonth() + 1}-${date.getDate()}`;
}

/** Write streak data to localStorage */
export function saveStreakToLocal(data: StreakData) {
  try {
    localStorage.setItem(STREAK_KEY, JSON.stringify({
      current: data.current_streak,
      longest: data.longest_streak,
      activeDays: data.streak_days,
    }));
  } catch {}
}

/** Read streak data from localStorage (home page format) */
export function readStreakFromLocal(): { current: number; longest: number; activeDays: string[] } {
  try {
    const raw = localStorage.getItem(STREAK_KEY);
    if (!raw) return { current: 0, longest: 0, activeDays: [] };
    return JSON.parse(raw);
  } catch {
    return { current: 0, longest: 0, activeDays: [] };
  }
}

/**
 * Call this whenever a user ATTEMPTS a question (correct or incorrect).
 * Gets userId from localStorage @user if not logged in via Supabase auth.
 */
export async function updateStreak() {
  const todayKey = toDateKey();
  const yesterdayKey = toDateKey(new Date(Date.now() - 86400000));

  // --- Get user id ---
  let userId: string | null = null;
  try {
    const { data: { user } } = await supabase.auth.getUser();
    userId = user?.id ?? null;
  } catch {}
  if (!userId) {
    try {
      const cached = localStorage.getItem('@user');
      if (cached) userId = JSON.parse(cached)?.id ?? null;
    } catch {}
  }
  if (!userId) return; // can't track without an id

  // --- Fetch existing row ---
  const { data: existing } = await supabase
    .from('user_streaks')
    .select('*')
    .eq('user_id', userId)
    .single();

  const prev: StreakData = existing
    ? {
        current_streak: existing.current_streak,
        longest_streak: existing.longest_streak,
        last_attempt_date: existing.last_attempt_date,
        streak_days: existing.streak_days ?? [],
      }
    : { current_streak: 0, longest_streak: 0, last_attempt_date: null, streak_days: [] };

  // --- Already attempted today? No change needed ---
  if (prev.streak_days.includes(todayKey)) {
    saveStreakToLocal(prev);
    return;
  }

  // --- Calculate new streak ---
  const wasYesterday = prev.last_attempt_date === yesterdayKey;
  const newCurrent = wasYesterday ? prev.current_streak + 1 : 1;
  const newLongest = Math.max(prev.longest_streak, newCurrent);
  const newDays = [...prev.streak_days, todayKey];

  const updated: StreakData = {
    current_streak: newCurrent,
    longest_streak: newLongest,
    last_attempt_date: todayKey,
    streak_days: newDays,
  };

  // --- Upsert to Supabase ---
  await supabase.from('user_streaks').upsert({
    user_id: userId,
    current_streak: newCurrent,
    longest_streak: newLongest,
    last_attempt_date: todayKey,
    streak_days: newDays,
    updated_at: new Date().toISOString(),
  }, { onConflict: 'user_id' });

  // --- Sync to localStorage ---
  saveStreakToLocal(updated);
}

/**
 * Load streak from Supabase into localStorage in the background.
 * Returns immediately, syncs in background (fire and forget).
 * This ensures instant UI with local data while keeping Supabase in sync.
 */
export async function syncStreakFromSupabase() {
  // Fire the sync in background without awaiting
  syncStreakInBackground();
}

/**
 * Internal function that actually performs the background sync.
 */
async function syncStreakInBackground() {
  let userId: string | null = null;
  try {
    const { data: { user } } = await supabase.auth.getUser();
    userId = user?.id ?? null;
  } catch {}
  if (!userId) {
    try {
      const cached = localStorage.getItem('@user');
      if (cached) userId = JSON.parse(cached)?.id ?? null;
    } catch {}
  }
  if (!userId) return;

  try {
    const { data } = await supabase
      .from('user_streaks')
      .select('*')
      .eq('user_id', userId)
      .single();

    if (data) {
      const remoteData: StreakData = {
        current_streak: data.current_streak,
        longest_streak: data.longest_streak,
        last_attempt_date: data.last_attempt_date,
        streak_days: data.streak_days ?? [],
      };

      // Only update if remote has newer/different data
      const local = readStreakFromLocal();
      if (
        local.current !== remoteData.current_streak ||
        local.longest !== remoteData.longest_streak ||
        JSON.stringify(local.activeDays) !== JSON.stringify(remoteData.streak_days)
      ) {
        saveStreakToLocal(remoteData);
        // Dispatch custom event so StreakCard can re-render if needed
        window.dispatchEvent(new CustomEvent('streakUpdated', { detail: remoteData }));
      }
    }
  } catch (err) {
    console.warn('Failed to sync streak from Supabase:', err);
  }
}