'use client';

import { useEffect, useState } from 'react';
import { AlignLeft } from 'lucide-react';

interface Heading {
  id: string;
  text: string;
  level: number;
}

interface TableOfContentsProps {
  headings: Heading[];
}

export default function TableOfContents({ headings }: TableOfContentsProps) {
  const [activeId, setActiveId] = useState<string>('');

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            setActiveId(entry.target.id);
          }
        });
      },
      { rootMargin: '0px 0px -80% 0px' },
    );

    headings.forEach((heading) => {
      const element = document.getElementById(heading.id);
      if (element) {
        observer.observe(element);
      }
    });

    return () => {
      headings.forEach((heading) => {
        const element = document.getElementById(heading.id);
        if (element) {
          observer.unobserve(element);
        }
      });
    };
  }, [headings]);

  if (headings.length === 0) return null;

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2 text-[var(--foreground-muted)] mb-4">
        <AlignLeft size={16} />
        <span className="text-sm font-bold uppercase tracking-wider">
          On this page
        </span>
      </div>
      <nav className="space-y-1">
        {headings.map((heading) => (
          <a
            key={heading.id}
            href={`#${heading.id}`}
            onClick={(e) => {
              e.preventDefault();
              document.getElementById(heading.id)?.scrollIntoView({
                behavior: 'smooth',
                block: 'start',
              });
              setActiveId(heading.id);
            }}
            className={`
              block text-sm py-1.5 transition-colors border-l-2 pl-4
              ${
                activeId === heading.id
                  ? 'border-[var(--accent)] text-[var(--accent)] font-medium'
                  : 'border-transparent text-[var(--foreground-muted)] hover:text-[var(--foreground)]'
              }
              ${heading.level === 3 ? 'ml-4' : ''}
            `}
          >
            {heading.text}
          </a>
        ))}
      </nav>
    </div>
  );
}
