import { createClient } from '@supabase/supabase-js';

// Replace with your actual Supabase project URL and anon key
const SUPABASE_URL = 'https://rzcizwacjexolkjjczbt.supabase.co';
const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJ6Y2l6d2FjamV4b2xrampjemJ0Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDU0MTA2ODMsImV4cCI6MjA2MDk4NjY4M30.I5TO7lLOuBwe6T5wllcx3FK_is0pammMtVw-oevfTws';

export const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);