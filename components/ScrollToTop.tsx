'use client';

import Link from 'next/link';

export default function ScrollToTop() {
  return (
    <Link
      href="#"
      onClick={(e) => {
        e.preventDefault();
        window.scrollTo({ top: 0, behavior: 'smooth' });
      }}
      className="hover:text-[var(--accent)] transition-colors"
    >
      Back to top ↑
    </Link>
  );
}
