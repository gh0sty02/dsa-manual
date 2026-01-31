'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { ChevronDown, ChevronRight, BookOpen } from 'lucide-react';
import { useState } from 'react';

export interface SidebarItem {
  title: string;
  slug: string;
  categories: string[];
}

interface SidebarProps {
  items: SidebarItem[];
  categories: { name: string; slug: string; count: number }[];
  onLinkClick?: () => void;
}

export default function Sidebar({
  items,
  categories,
  onLinkClick,
}: SidebarProps) {
  const pathname = usePathname();

  // Group items by category - this logic must be BEFORE component state logic that uses it
  const itemsByCategory = categories.map((category) => ({
    ...category,
    items: items.filter((item) => {
      // Check if any of the item's categories matches this category
      return item.categories.some(
        (cat) =>
          cat.toLowerCase().replace(/\s+/g, '-').replace(/&/g, 'and') ===
          category.slug,
      );
    }),
  }));

  // Determine active category based on current path
  const activeCategory = itemsByCategory.find((cat) =>
    cat.items.some((item) => pathname === `/${item.slug}`),
  );

  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(
    activeCategory ? new Set([activeCategory.slug]) : new Set(),
  );

  const toggleCategory = (categorySlug: string) => {
    setExpandedCategories((prev) => {
      const next = new Set(prev);
      if (next.has(categorySlug)) {
        next.delete(categorySlug);
      } else {
        next.add(categorySlug);
      }
      return next;
    });
  };

  return (
    <aside className="w-full">
      <nav className="space-y-1">
        {itemsByCategory.map((category) => {
          const isExpanded = expandedCategories.has(category.slug);
          const hasActiveItem = category.items.some(
            (item) => pathname === `/${item.slug}`,
          );

          if (category.items.length === 0) return null;

          return (
            <div key={category.slug} className="mb-2">
              {/* Category Header */}
              <button
                onClick={() => toggleCategory(category.slug)}
                className={`
                  w-full flex items-center justify-between px-3 py-2 text-left
                  text-sm font-semibold rounded-lg transition-colors duration-150
                  ${
                    hasActiveItem
                      ? 'text-[var(--accent)] bg-[var(--accent-light)]'
                      : 'text-[var(--foreground)] hover:bg-[var(--surface-hover)]'
                  }
                `}
              >
                <span className="flex items-center gap-2">
                  <BookOpen size={16} className="opacity-60" />
                  {category.name}
                </span>
                <span className="flex items-center gap-2">
                  <span className="text-xs text-[var(--foreground-muted)] bg-[var(--surface)] px-1.5 py-0.5 rounded">
                    {category.count}
                  </span>
                  {isExpanded ? (
                    <ChevronDown size={16} className="opacity-60" />
                  ) : (
                    <ChevronRight size={16} className="opacity-60" />
                  )}
                </span>
              </button>

              {/* Category Items */}
              {isExpanded && (
                <div className="mt-1 ml-3 pl-3 border-l border-[var(--border)] space-y-0.5">
                  {category.items.map((item) => {
                    const isActive = pathname === `/${item.slug}`;
                    return (
                      <Link
                        key={item.slug}
                        href={`/${item.slug}`}
                        onClick={onLinkClick}
                        className={`
                          block px-3 py-1.5 text-sm rounded-md transition-all duration-150
                          ${
                            isActive
                              ? 'text-[var(--accent)] bg-[var(--accent-light)] font-medium'
                              : 'text-[var(--foreground-muted)] hover:text-[var(--foreground)] hover:bg-[var(--surface-hover)]'
                          }
                        `}
                      >
                        {item.title
                          .replace(/^\d+-/, '')
                          .replace(/-pattern-explained$/, '')
                          .replace(/-/g, ' ')
                          .replace(/\b\w/g, (l) => l.toUpperCase())
                          .replace('Dynamic Programming', 'DP')
                          .trim()}
                      </Link>
                    );
                  })}
                </div>
              )}
            </div>
          );
        })}
      </nav>
    </aside>
  );
}
