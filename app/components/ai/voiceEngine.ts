'use client';
// ─── VOICE ENGINE ─────────────────────────────────────────────────────────────
// Handles:
//   - Speech-to-text (Web Speech API, free)
//   - Text-to-speech (Web Speech Synthesis API, free, with prosody hints)
//   - Voice activity detection (VAD)
//   - Humanized speech: pauses, rate, pitch per emotion

import { ProsodyHints } from './emotionEngine';

// ─── SPEECH TO TEXT ───────────────────────────────────────────────────────────

export interface STTResult {
  transcript: string;
  confidence: number;
  isFinal: boolean;
}

export class SpeechRecognitionEngine {
  private recognition: any = null;
  private isListening = false;

  constructor(
    private onResult: (result: STTResult) => void,
    private onEnd: () => void,
    private onError: (err: string) => void
  ) {
    const SR = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SR) {
      console.warn('SpeechRecognition not supported');
      return;
    }
    this.recognition = new SR();
    this.recognition.continuous = true;       // Keep listening until explicitly stopped
    this.recognition.interimResults = true;    // Show live transcript
    this.recognition.lang = 'en-IN';
    this.recognition.maxAlternatives = 1;

    this.recognition.onresult = (event: any) => {
      const result = event.results[event.results.length - 1];
      const transcript = result[0].transcript;
      const confidence = result[0].confidence;
      const isFinal = result.isFinal;
      this.onResult({ transcript, confidence, isFinal });
    };

    this.recognition.onend = () => {
      this.isListening = false;
      this.onEnd();
    };

    this.recognition.onerror = (event: any) => {
      this.isListening = false;
      this.onError(event.error);
    };
  }

  start() {
    if (this.recognition && !this.isListening) {
      this.recognition.start();
      this.isListening = true;
    }
  }

  stop() {
    if (this.recognition && this.isListening) {
      this.recognition.stop();
      this.isListening = false;
    }
  }

  get active() { return this.isListening; }

  static isSupported(): boolean {
    return typeof window !== 'undefined' && (
      !!(window as any).SpeechRecognition || !!(window as any).webkitSpeechRecognition
    );
  }
}

// ─── TEXT TO SPEECH (Web Speech Synthesis — 100% Free) ───────────────────────

export class SpeechSynthesisEngine {
  private utterance: SpeechSynthesisUtterance | null = null;
  private isSpeaking = false;

  speak(
    text: string,
    prosody: ProsodyHints,
    voiceStyle: string,
    onStart?: () => void,
    onEnd?: () => void
  ) {
    if (typeof window === 'undefined' || !window.speechSynthesis) return;

    // Stop any current speech
    this.stop();

    // Clean text for speech (remove markdown, code blocks, etc.)
    const cleanText = this.cleanForSpeech(text);

    // Split into natural sentences and add pauses
    const sentences = this.splitIntoSentences(cleanText);
    this.speakSequentially(sentences, prosody, voiceStyle, onStart, onEnd);
  }

  private speakSequentially(
    sentences: string[],
    prosody: ProsodyHints,
    voiceStyle: string,
    onStart?: () => void,
    onEnd?: () => void,
    index = 0
  ) {
    if (index >= sentences.length) {
      this.isSpeaking = false;
      onEnd?.();
      return;
    }

    const sentence = sentences[index];
    if (!sentence.trim()) {
      this.speakSequentially(sentences, prosody, voiceStyle, onStart, onEnd, index + 1);
      return;
    }

    this.utterance = new SpeechSynthesisUtterance(sentence);

    // Prosody settings
    const rateMap = { slow: 0.85, medium: 1.0, fast: 1.15 };
    const pitchMap = { low: 0.9, medium: 1.0, high: 1.1 };
    this.utterance.rate = rateMap[prosody.rate];
    this.utterance.pitch = pitchMap[prosody.pitch];
    this.utterance.volume = 1;

    // Pick voice based on style
    this.utterance.voice = this.pickVoice(voiceStyle || 'warm');

    if (index === 0) {
      this.isSpeaking = true;
      onStart?.();
    }

    this.utterance.onend = () => {
      const pause = index === 0
        ? prosody.pauseAfterGreeting
        : prosody.pauseBetweenSentences;

      setTimeout(() => {
        this.speakSequentially(sentences, prosody, voiceStyle, onStart, onEnd, index + 1);
      }, pause);
    };

    window.speechSynthesis.speak(this.utterance);
  }

  private pickVoice(style: string): SpeechSynthesisVoice | null {
    const voices = window.speechSynthesis.getVoices();
    if (!voices.length) return null;

    const preferredVoice: Record<string, string[]> = {
      confident: ['Google UK English Male', 'Alex', 'Daniel'],
      warm:      ['Google UK English Female', 'Samantha', 'Karen'],
      calm:      ['Google US English', 'Tom', 'Daniel'],
      energetic: ['Zira', 'Google US English', 'Victoria'],
    };

    const preferred = preferredVoice[style] || preferredVoice.warm;

    for (const name of preferred) {
      const match = voices.find(v => v.name.includes(name));
      if (match) return match;
    }

    // Fallback: pick first English voice
    return voices.find(v => v.lang.startsWith('en')) || voices[0];
  }

  stop() {
    if (typeof window !== 'undefined' && window.speechSynthesis) {
      window.speechSynthesis.cancel();
    }
    this.isSpeaking = false;
  }

  get speaking() { return this.isSpeaking; }

  private cleanForSpeech(text: string): string {
    return text
      .replace(/```[\s\S]*?```/g, 'code block')      // code blocks
      .replace(/`[^`]+`/g, '')                        // inline code
      .replace(/#{1,6}\s/g, '')                       // headers
      .replace(/\*\*([^*]+)\*\*/g, '$1')              // bold
      .replace(/\*([^*]+)\*/g, '$1')                  // italic
      .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')        // links
      .replace(/^\s*[-*+]\s/gm, '')                   // list bullets
      .replace(/\n{2,}/g, '. ')                       // double newlines → pause
      .replace(/\n/g, ' ')
      .trim();
  }

  private splitIntoSentences(text: string): string[] {
    // Split on sentence boundaries while keeping natural chunks
    return text
      .split(/(?<=[.!?])\s+/)
      .filter(s => s.trim().length > 0);
  }
}

// ─── VOICE ACTIVITY DETECTOR (simple amplitude-based) ────────────────────────
// Returns a hook-friendly class that detects if user is speaking

export class VoiceActivityDetector {
  private audioCtx: AudioContext | null = null;
  private analyser: AnalyserNode | null = null;
  private stream: MediaStream | null = null;
  private animationFrame: number | null = null;

  async start(onActivity: (active: boolean, level: number) => void): Promise<void> {
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      this.audioCtx = new AudioContext();
      const source = this.audioCtx.createMediaStreamSource(this.stream);
      this.analyser = this.audioCtx.createAnalyser();
      this.analyser.fftSize = 512;
      source.connect(this.analyser);

      const dataArray = new Uint8Array(this.analyser.frequencyBinCount);
      let silenceCount = 0;
      let wasActive = false;

      const tick = () => {
        this.analyser!.getByteFrequencyData(dataArray);
        const avg = dataArray.reduce((a, b) => a + b, 0) / dataArray.length;
        const level = avg / 128; // normalized 0–1
        const active = avg > 20; // threshold

        if (active) {
          silenceCount = 0;
          if (!wasActive) {
            wasActive = true;
            onActivity(true, level);
          } else {
            onActivity(true, level);
          }
        } else {
          silenceCount++;
          if (wasActive && silenceCount > 30) { // ~500ms of silence
            wasActive = false;
            onActivity(false, 0);
          }
        }

        this.animationFrame = requestAnimationFrame(tick);
      };

      tick();
    } catch (err) {
      console.error('VAD error:', err);
    }
  }

  stop() {
    if (this.animationFrame) cancelAnimationFrame(this.animationFrame);
    this.stream?.getTracks().forEach(t => t.stop());
    this.audioCtx?.close();
  }
}

export const voiceEngine = new SpeechSynthesisEngine();