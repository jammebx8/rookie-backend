'use client';
// ─── SIDEBAR COMPONENT ───────────────────────────────────────────────────────
// Extracted from page.tsx so chat state changes don't re-render the sidebar.
// Uses React.memo + stable prop references to prevent unnecessary renders.

import React, { memo } from 'react';
import Image from 'next/image';
import { motion, AnimatePresence } from 'framer-motion';

interface Conversation {
  id: string;
  title: string;
  updated_at: string;
  persona_id: number;
}

interface UserProfile {
  id: string;
  email: string;
  name?: string | null;
  avatar_url?: string | null;
}

interface Persona {
  id: number;
  name: string;
  avatar: string;
  accent: string;
}

interface SidebarProps {
  conversations: Conversation[];
  activeConversationId: string | null;
  selectedPersona: Persona;
  userProfile: UserProfile | null;
  historyOpen: boolean;
  showPersonaModal: boolean;
  onNewChat: () => void;
  onLoadConversation: (id: string) => void;
  onDeleteConversation: (id: string, e: React.MouseEvent) => void;
  onToggleHistory: () => void;
  onOpenPersonaModal: () => void;
  onCollapse: () => void;
}

// ─── ICONS ────────────────────────────────────────────────────────────────────

const PlusIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/>
  </svg>
);

const ChevronIcon = ({ open }: { open: boolean }) => (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" style={{ transform: open ? 'rotate(180deg)' : 'rotate(0deg)', transition: 'transform 0.2s' }}>
    <polyline points="6 9 12 15 18 9"/>
  </svg>
);

const TrashIcon = () => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/>
  </svg>
);

const CollapseIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="15 18 9 12 15 6"/>
  </svg>
);

// ─── HELPERS ─────────────────────────────────────────────────────────────────

function formatDate(dateStr: string): string {
  const d = new Date(dateStr);
  const now = new Date();
  const diff = now.getTime() - d.getTime();
  const days = Math.floor(diff / 86400000);
  if (days === 0) return 'Today';
  if (days === 1) return 'Yesterday';
  if (days < 7) return `${days}d ago`;
  return d.toLocaleDateString('en', { month: 'short', day: 'numeric' });
}

// ─── SIDEBAR ─────────────────────────────────────────────────────────────────

const Sidebar = memo(function Sidebar({
  conversations,
  activeConversationId,
  selectedPersona,
  userProfile,
  historyOpen,
  showPersonaModal,
  onNewChat,
  onLoadConversation,
  onDeleteConversation,
  onToggleHistory,
  onOpenPersonaModal,
  onCollapse,
}: SidebarProps) {
  const getUserDisplayName = () =>
    userProfile?.name || userProfile?.email?.split('@')[0] || 'User';
  const getAvatarInitial = () =>
    (userProfile?.name?.[0] || userProfile?.email?.[0] || 'U').toUpperCase();

  return (
    <div className="flex flex-col h-full">
      {/* Logo & collapse */}
      <div className="flex items-center justify-between px-4 pb-4 pt-2">
        <img src="/fridaylogo.jpg" alt="Friday Logo" className="w-32 h-auto object-contain" />
        <button onClick={onCollapse} className="text-[#666] hover:text-white transition-colors p-1">
          <CollapseIcon />
        </button>
      </div>

      {/* New Chat */}
      <div className="px-3 pb-3">
        <button
          onClick={onNewChat}
          className="w-full flex items-center gap-2.5 px-3 py-2.5 rounded-xl text-[#ccc] hover:text-white hover:bg-white/5 transition-all text-sm font-medium border border-transparent hover:border-white/10"
        >
          <PlusIcon />
          <span>New Chat</span>
        </button>
      </div>

      {/* Persona selector */}
      <div className="px-3 pb-3">
        <button
          onClick={onOpenPersonaModal}
          className="w-full flex items-center gap-3 px-3 py-2.5 rounded-xl hover:bg-white/5 transition-all border border-white/10"
        >
          <div
            className="w-8 h-8 rounded-lg text-lg flex items-center justify-center"
            style={{ background: `${selectedPersona.accent}22` }}
          >
            {selectedPersona.avatar}
          </div>
          <div className="flex-1 text-left">
            <p className="text-white text-xs font-semibold">{selectedPersona.name}</p>
            <p className="text-[#555] text-[10px]">Active persona</p>
          </div>
          <ChevronIcon open={showPersonaModal} />
        </button>
      </div>

      <div className="px-3 pb-2">
        <div className="h-px bg-white/5" />
      </div>

      {/* Conversation history */}
      <div className="flex-1 overflow-y-auto px-3" style={{ msOverflowStyle: 'none', scrollbarWidth: 'none' }}>
        <button
          onClick={onToggleHistory}
          className="w-full flex items-center justify-between px-2 py-2 text-[#666] hover:text-[#999] transition-colors"
        >
          <span className="text-[11px] font-semibold uppercase tracking-widest">History</span>
          <ChevronIcon open={historyOpen} />
        </button>

        <AnimatePresence>
          {historyOpen && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="overflow-hidden"
            >
              <div className="space-y-0.5 pb-4">
                {conversations.length === 0 && (
                  <p className="text-[#444] text-xs px-2 py-4 text-center">No conversations yet</p>
                )}
                {conversations.map(conv => (
                  <motion.div
                    key={conv.id}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    className={`group flex items-center gap-2 px-2 py-2 rounded-lg cursor-pointer transition-all ${
                      activeConversationId === conv.id
                        ? 'bg-white/8 text-white'
                        : 'text-[#666] hover:text-[#ccc] hover:bg-white/4'
                    }`}
                    onClick={() => onLoadConversation(conv.id)}
                  >
                    <div className="flex-1 min-w-0">
                      <p className="text-xs truncate font-medium">{conv.title}</p>
                      <p className="text-[10px] text-[#444] mt-0.5">{formatDate(conv.updated_at)}</p>
                    </div>
                    <button
                      onClick={(e) => onDeleteConversation(conv.id, e)}
                      className="opacity-0 group-hover:opacity-100 text-[#444] hover:text-red-400 transition-all p-0.5"
                    >
                      <TrashIcon />
                    </button>
                  </motion.div>
                ))}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* User profile */}
      <div className="px-3 py-3 border-t border-white/5">
        <div className="flex items-center gap-3 px-2 py-2">
          <div className="w-8 h-8 rounded-full bg-white/10 overflow-hidden flex-shrink-0">
            {userProfile?.avatar_url ? (
              <Image src={userProfile.avatar_url} alt="Avatar" width={32} height={32} className="object-cover" />
            ) : (
              <div className="w-full h-full flex items-center justify-center text-white text-xs font-bold">
                {getAvatarInitial()}
              </div>
            )}
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-white text-xs font-medium truncate">{getUserDisplayName()}</p>
            <p className="text-[#555] text-[10px] truncate">{userProfile?.email}</p>
          </div>
        </div>
      </div>
    </div>
  );
});

export default Sidebar;