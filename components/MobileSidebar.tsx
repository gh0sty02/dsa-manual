'use client';

import { useState, useEffect } from 'react';
import { createPortal } from 'react-dom';
import { Menu, X, Code2 } from 'lucide-react';
import Sidebar, { SidebarItem } from './Sidebar';
import Link from 'next/link';

interface MobileSidebarProps {
  items: SidebarItem[];
  categories: { name: string; slug: string; count: number }[];
}

export default function MobileSidebar({
  items,
  categories,
}: MobileSidebarProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  // Prevent scrolling when sidebar is open
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }
  }, [isOpen]);

  const SidebarContent = (
    <div className="fixed inset-0 z-[100] flex">
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/50 backdrop-blur-sm transition-opacity"
        onClick={() => setIsOpen(false)}
      />

      {/* Sidebar Content */}
      <div className="relative w-[85%] max-w-xs bg-[var(--background)] h-full shadow-2xl flex flex-col border-r border-[var(--border)] overflow-hidden">
        <div className="p-4 border-b border-[var(--border)] flex items-center justify-between bg-[var(--background)] flex-shrink-0">
          <Link
            href="/"
            className="flex items-center gap-3"
            onClick={() => setIsOpen(false)}
          >
            <div className="p-1.5 rounded-lg bg-[var(--accent)] text-white">
              <Code2 size={18} />
            </div>
            <span className="font-display font-bold text-lg text-[var(--foreground)]">
              DSA Manual
            </span>
          </Link>
          <button
            onClick={() => setIsOpen(false)}
            className="p-1.5 rounded-md hover:bg-[var(--surface-hover)] text-[var(--foreground-muted)] hover:text-[var(--foreground)] transition-colors"
          >
            <X size={20} />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-2 py-4 bg-[var(--background)]">
          <Sidebar
            items={items}
            categories={categories}
            onLinkClick={() => setIsOpen(false)}
          />
        </div>
      </div>
    </div>
  );

  return (
    <div className="lg:hidden flex items-center">
      <button
        onClick={() => setIsOpen(true)}
        className="p-2 -ml-2 mr-2 text-[var(--foreground-muted)] hover:text-[var(--foreground)] transition-colors"
        aria-label="Open menu"
      >
        <Menu size={24} />
      </button>

      {/* Render via Portal to avoid stacking context issues */}
      {isOpen && mounted && createPortal(SidebarContent, document.body)}
    </div>
  );
}
