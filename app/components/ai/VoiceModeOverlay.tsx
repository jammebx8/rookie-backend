'use client';
// ─── VOICE MODE OVERLAY ───────────────────────────────────────────────────────
// Full-screen immersive voice conversation UI.
// Shows: animated orb (AI speaking), waveform (user speaking), transcript.

import React, { useEffect, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface VoiceModeOverlayProps {
  isOpen: boolean;
  personaName: string;
  personaAvatar: string;
  accentColor: string;
  isAISpeaking: boolean;
  isUserSpeaking: boolean;
  voiceLevel: number;         // 0–1, user mic amplitude
  liveTranscript: string;     // live STT transcript
  lastAIMessage: string;      // last thing AI said
  onClose: () => void;
  onStartListening: () => void;
  onStopListening: () => void;
  isListening: boolean;
}

// ─── ANIMATED ORB (AI State Visualizer) ───────────────────────────────────────

const AIOrb = ({
  accentColor,
  isAISpeaking,
  isListening,
}: {
  accentColor: string;
  isAISpeaking: boolean;
  isListening: boolean;
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef<number>();
  const phaseRef = useRef(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d')!;
    const size = canvas.width;
    const cx = size / 2;
    const cy = size / 2;
    const r = size * 0.3;

    const draw = () => {
      phaseRef.current += isAISpeaking ? 0.05 : 0.015;
      const phase = phaseRef.current;

      ctx.clearRect(0, 0, size, size);

      // Glow ring
      const gradient = ctx.createRadialGradient(cx, cy, r * 0.5, cx, cy, r * 1.6);
      gradient.addColorStop(0, accentColor + '44');
      gradient.addColorStop(1, 'transparent');
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, size, size);

      // Blob shape
      ctx.beginPath();
      const points = 8;
      for (let i = 0; i <= points * 10; i++) {
        const angle = (i / (points * 10)) * Math.PI * 2;
        const noise = isAISpeaking
          ? 1 + 0.15 * Math.sin(phase * 3 + angle * 3) * Math.cos(phase * 2 + angle * 5)
          : 1 + 0.04 * Math.sin(phase + angle * 2);
        const radius = r * noise;
        const x = cx + radius * Math.cos(angle);
        const y = cy + radius * Math.sin(angle);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.closePath();

      const blobGrad = ctx.createRadialGradient(cx - r * 0.3, cy - r * 0.3, 0, cx, cy, r * 1.2);
      blobGrad.addColorStop(0, accentColor + 'ff');
      blobGrad.addColorStop(0.5, accentColor + 'cc');
      blobGrad.addColorStop(1, accentColor + '44');
      ctx.fillStyle = blobGrad;
      ctx.fill();

      // Inner glow
      ctx.beginPath();
      ctx.arc(cx - r * 0.2, cy - r * 0.2, r * 0.25, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(255,255,255,0.15)';
      ctx.fill();

      animRef.current = requestAnimationFrame(draw);
    };

    draw();
    return () => { if (animRef.current) cancelAnimationFrame(animRef.current); };
  }, [accentColor, isAISpeaking]);

  return (
    <div className="relative">
      <canvas
        ref={canvasRef}
        width={240}
        height={240}
        className="drop-shadow-2xl"
        style={{ filter: `drop-shadow(0 0 30px ${accentColor}66)` }}
      />
      {isListening && !isAISpeaking && (
        <motion.div
          className="absolute inset-0 rounded-full border-2 border-white/20"
          animate={{ scale: [1, 1.2, 1], opacity: [0.5, 0, 0.5] }}
          transition={{ duration: 2, repeat: Infinity }}
        />
      )}
    </div>
  );
};

// ─── USER WAVEFORM ────────────────────────────────────────────────────────────

const UserWaveform = ({ level, isActive }: { level: number; isActive: boolean }) => {
  const bars = 20;
  return (
    <div className="flex items-center justify-center gap-[3px] h-12">
      {Array.from({ length: bars }).map((_, i) => {
        const center = bars / 2;
        const distFromCenter = Math.abs(i - center) / center;
        const baseHeight = (1 - distFromCenter * 0.6) * 0.3;
        const activeHeight = isActive
          ? baseHeight + level * (0.7 - distFromCenter * 0.4) * Math.random()
          : baseHeight * 0.3;
        return (
          <motion.div
            key={i}
            className="rounded-full"
            style={{
              width: 3,
              background: isActive ? '#ffffff' : '#444',
            }}
            animate={{ height: Math.max(4, activeHeight * 48) }}
            transition={{ duration: 0.1, ease: 'easeOut' }}
          />
        );
      })}
    </div>
  );
};

// ─── MAIN OVERLAY ─────────────────────────────────────────────────────────────

export default function VoiceModeOverlay({
  isOpen,
  personaName,
  personaAvatar,
  accentColor,
  isAISpeaking,
  isUserSpeaking,
  voiceLevel,
  liveTranscript,
  lastAIMessage,
  onClose,
  onStartListening,
  onStopListening,
  isListening,
}: VoiceModeOverlayProps) {
  const [displayText, setDisplayText] = useState('');

  useEffect(() => {
    if (isAISpeaking && lastAIMessage) {
      // Typewriter effect for AI speech display
      setDisplayText('');
      let i = 0;
      const words = lastAIMessage.split(' ');
      const interval = setInterval(() => {
        if (i < words.length) {
          setDisplayText(prev => (prev ? prev + ' ' : '') + words[i]);
          i++;
        } else {
          clearInterval(interval);
        }
      }, 120);
      return () => clearInterval(interval);
    }
  }, [lastAIMessage, isAISpeaking]);

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-50 flex flex-col items-center justify-center"
          style={{
            background: `radial-gradient(ellipse at center, ${accentColor}11 0%, #000000 70%)`,
          }}
        >
          {/* Close */}
          <button
            onClick={onClose}
            className="absolute top-6 right-6 text-[#555] hover:text-white transition-colors p-2"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
            </svg>
          </button>

          {/* Persona name */}
          <motion.div
            initial={{ y: -20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.2 }}
            className="mb-2 text-center"
          >
            <p className="text-[#555] text-xs uppercase tracking-widest">Talking with</p>
            <p className="text-white text-xl font-bold mt-1">{personaAvatar} {personaName}</p>
          </motion.div>

          {/* Orb */}
          <motion.div
            initial={{ scale: 0.5, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ delay: 0.3, type: 'spring', stiffness: 200 }}
            className="my-8"
          >
            <AIOrb
              accentColor={accentColor}
              isAISpeaking={isAISpeaking}
              isListening={isListening}
            />
          </motion.div>

          {/* Status label */}
          <motion.p
            className="text-[#555] text-xs uppercase tracking-widest mb-6"
            animate={{ opacity: [0.5, 1, 0.5] }}
            transition={{ duration: 2, repeat: Infinity }}
          >
            {isAISpeaking ? `${personaName} is speaking...` : isListening ? 'Listening...' : 'Tap mic to speak'}
          </motion.p>

          {/* AI speech text */}
          <div className="w-full max-w-sm px-6 mb-8 min-h-[60px] text-center">
            <AnimatePresence mode="wait">
              {isAISpeaking && displayText && (
                <motion.p
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  className="text-[#ccc] text-sm leading-relaxed"
                >
                  {displayText}
                </motion.p>
              )}
              {!isAISpeaking && liveTranscript && (
                <motion.p
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="text-[#888] text-sm italic"
                >
                  "{liveTranscript}"
                </motion.p>
              )}
            </AnimatePresence>
          </div>

          {/* User waveform */}
          <div className="mb-8">
            <UserWaveform level={voiceLevel} isActive={isListening && isUserSpeaking} />
          </div>

          {/* Mic button */}
          <motion.button
            whileTap={{ scale: 0.92 }}
            onClick={isListening ? onStopListening : onStartListening}
            className={`w-16 h-16 rounded-full flex items-center justify-center transition-all border-2 ${
              isListening
                ? 'border-red-500 bg-red-500/20'
                : 'border-white/20 bg-white/5 hover:bg-white/10'
            }`}
            style={isListening ? { boxShadow: '0 0 30px rgba(239,68,68,0.4)' } : {}}
          >
            {isListening ? (
              <motion.div
                animate={{ scale: [1, 1.2, 1] }}
                transition={{ duration: 1, repeat: Infinity }}
              >
                <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor" className="text-red-400">
                  <rect x="4" y="4" width="16" height="16" rx="2"/>
                </svg>
              </motion.div>
            ) : (
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-white">
                <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
                <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                <line x1="12" y1="19" x2="12" y2="23"/>
                <line x1="8" y1="23" x2="16" y2="23"/>
              </svg>
            )}
          </motion.button>

          <p className="text-[#333] text-[10px] mt-4">
            {isListening ? 'Tap to stop' : 'Tap to speak'}
          </p>
        </motion.div>
      )}
    </AnimatePresence>
  );
}