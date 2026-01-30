import Link from 'next/link';
import { getAllPosts, getCategories } from '@/lib/api';
import Sidebar from '@/components/Sidebar';
import {
  BookOpen,
  Code2,
  Layers,
  Sparkles,
  ArrowRight,
  Github,
} from 'lucide-react';

export default function Home() {
  const posts = getAllPosts();
  const categories = getCategories();

  // Get featured patterns (first 3 or specific ones)
  const featuredSlugs = [
    '01-two-pointers-pattern-explained',
    '03-sliding-window-pattern-explained',
    '17-modified-binary-search-pattern-explained',
  ];
  const featured = posts
    .filter((p) => featuredSlugs.includes(p.slug))
    .slice(0, 3);

  // Prepare sidebar items
  const sidebarItems = posts.map((post) => ({
    title: post.title,
    slug: post.slug,
    category: post.category || 'Other',
  }));

  return (
    <div className="min-h-screen bg-[var(--background)]">
      {/* Header */}
      <header className="sticky top-0 z-50 backdrop-blur-md bg-[var(--background)]/80 border-b border-[var(--border)]">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <Link href="/" className="flex items-center gap-3 group">
              <div className="p-2 rounded-lg bg-[var(--accent)] text-white group-hover:scale-105 transition-transform">
                <Code2 size={20} />
              </div>
              <span className="font-display font-bold text-lg text-[var(--foreground)]">
                DSA Patterns
              </span>
            </Link>

            <nav className="hidden md:flex items-center gap-6">
              <Link
                href="https://github.com/gh0sty02/dsa-patterns"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 text-sm text-[var(--foreground-muted)] hover:text-[var(--foreground)] transition-colors"
              >
                <Github size={18} />
                GitHub
              </Link>
            </nav>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="flex gap-8">
          {/* Sidebar */}
          <div className="hidden lg:block w-72 flex-shrink-0">
            <div className="sticky top-24">
              <Sidebar items={sidebarItems} categories={categories} />
            </div>
          </div>

          {/* Main Content */}
          <main className="flex-1 min-w-0">
            {/* Hero Section */}
            <section className="mb-12 animate-slide-up">
              <div className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-[var(--accent)] to-blue-700 p-8 md:p-12">
                <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PGRlZnM+PHBhdHRlcm4gaWQ9ImdyaWQiIHdpZHRoPSI2MCIgaGVpZ2h0PSI2MCIgcGF0dGVyblVuaXRzPSJ1c2VyU3BhY2VPblVzZSI+PHBhdGggZD0iTSAxMCAwIEwgMCAwIDAgMTAiIGZpbGw9Im5vbmUiIHN0cm9rZT0icmdiYSgyNTUsMjU1LDI1NSwwLjEpIiBzdHJva2Utd2lkdGg9IjEiLz48L3BhdHRlcm4+PC9kZWZzPjxyZWN0IHdpZHRoPSIxMDAlIiBoZWlnaHQ9IjEwMCUiIGZpbGw9InVybCgjZ3JpZCkiLz48L3N2Zz4=')] opacity-50"></div>
                <div className="relative z-10">
                  <div className="flex items-center gap-2 text-white/80 text-sm font-medium mb-4">
                    <Sparkles size={16} />
                    Master DSA Patterns
                  </div>
                  <h1 className="font-display text-3xl md:text-4xl lg:text-5xl font-extrabold text-white mb-4 tracking-tight">
                    Data Structures &<br />
                    Algorithms Patterns
                  </h1>
                  <p className="text-white/80 text-lg max-w-xl mb-6">
                    A comprehensive collection of {posts.length} DSA patterns to
                    help you ace coding interviews. Learn the underlying
                    patterns, not just individual problems.
                  </p>
                  <div className="flex flex-wrap gap-4">
                    <Link
                      href={`/${posts[0]?.slug || ''}`}
                      className="inline-flex items-center gap-2 px-5 py-2.5 bg-white text-[var(--accent)] font-semibold rounded-lg hover:bg-white/90 transition-colors"
                    >
                      Start Learning
                      <ArrowRight size={18} />
                    </Link>
                  </div>
                </div>
              </div>
            </section>

            {/* Stats */}
            <section className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-12 animate-slide-up stagger-1">
              <div className="p-4 rounded-xl bg-[var(--surface)] border border-[var(--border)]">
                <div className="text-2xl font-bold text-[var(--foreground)]">
                  {posts.length}
                </div>
                <div className="text-sm text-[var(--foreground-muted)]">
                  Patterns
                </div>
              </div>
              <div className="p-4 rounded-xl bg-[var(--surface)] border border-[var(--border)]">
                <div className="text-2xl font-bold text-[var(--foreground)]">
                  {categories.length}
                </div>
                <div className="text-sm text-[var(--foreground-muted)]">
                  Categories
                </div>
              </div>
              <div className="p-4 rounded-xl bg-[var(--surface)] border border-[var(--border)]">
                <div className="text-2xl font-bold text-[var(--foreground)]">
                  100+
                </div>
                <div className="text-sm text-[var(--foreground-muted)]">
                  Examples
                </div>
              </div>
              <div className="p-4 rounded-xl bg-[var(--surface)] border border-[var(--border)]">
                <div className="text-2xl font-bold text-[var(--foreground)]">
                  Free
                </div>
                <div className="text-sm text-[var(--foreground-muted)]">
                  Forever
                </div>
              </div>
            </section>

            {/* Featured Patterns */}
            {featured.length > 0 && (
              <section className="mb-12">
                <div className="flex items-center justify-between mb-6">
                  <h2 className="font-display text-xl font-bold text-[var(--foreground)]">
                    Featured Patterns
                  </h2>
                </div>
                <div className="grid gap-4 md:grid-cols-3">
                  {featured.map((post, index) => (
                    <Link
                      key={post.slug}
                      href={`/${post.slug}`}
                      className={`group relative p-6 rounded-xl bg-[var(--surface)] border border-[var(--border)] hover:border-[var(--accent)] hover:shadow-lg transition-all duration-200 animate-slide-up stagger-${index + 2}`}
                    >
                      <div className="flex items-center gap-2 mb-3">
                        <span className="text-xs font-medium px-2 py-1 rounded-full bg-[var(--accent-light)] text-[var(--accent)]">
                          {post.category}
                        </span>
                      </div>
                      <h3 className="font-display font-bold text-lg text-[var(--foreground)] group-hover:text-[var(--accent)] transition-colors mb-2 line-clamp-2">
                        {post.title
                          .replace(/^\d+-/, '')
                          .replace(/-pattern-explained$/, '')
                          .replace(/-/g, ' ')}
                      </h3>
                      <div className="flex items-center gap-1 text-sm text-[var(--foreground-muted)] group-hover:text-[var(--accent)]">
                        Read pattern{' '}
                        <ArrowRight
                          size={14}
                          className="group-hover:translate-x-1 transition-transform"
                        />
                      </div>
                    </Link>
                  ))}
                </div>
              </section>
            )}

            {/* All Patterns */}
            <section>
              <div className="flex items-center justify-between mb-6">
                <h2 className="font-display text-xl font-bold text-[var(--foreground)]">
                  All Patterns
                </h2>
                <span className="text-sm text-[var(--foreground-muted)]">
                  {posts.length} patterns
                </span>
              </div>
              <div className="space-y-2">
                {posts.map((post, index) => (
                  <Link
                    key={post.slug}
                    href={`/${post.slug}`}
                    className="group flex items-center justify-between p-4 rounded-lg bg-[var(--surface)] border border-[var(--border)] hover:border-[var(--accent)] hover:bg-[var(--surface-hover)] transition-all duration-150"
                    style={{
                      animationDelay: `${Math.min(index * 0.02, 0.3)}s`,
                    }}
                  >
                    <div className="flex items-center gap-4 min-w-0">
                      <div className="flex-shrink-0 w-8 h-8 rounded-lg bg-[var(--accent-light)] text-[var(--accent)] flex items-center justify-center">
                        <BookOpen size={16} />
                      </div>
                      <div className="min-w-0">
                        <h3 className="font-medium text-[var(--foreground)] group-hover:text-[var(--accent)] transition-colors truncate">
                          {post.title}
                        </h3>
                        <div className="flex items-center gap-2 text-xs text-[var(--foreground-muted)]">
                          <span>{post.category}</span>
                          <span>•</span>
                          <span>{post.difficulty}</span>
                        </div>
                      </div>
                    </div>
                    <ArrowRight
                      size={18}
                      className="flex-shrink-0 text-[var(--foreground-muted)] group-hover:text-[var(--accent)] group-hover:translate-x-1 transition-all"
                    />
                  </Link>
                ))}
              </div>
            </section>
          </main>
        </div>
      </div>

      {/* Footer */}
      <footer className="mt-20 border-t border-[var(--border)] py-8">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex flex-col md:flex-row items-center justify-between gap-4 text-sm text-[var(--foreground-muted)]">
            <div className="flex items-center gap-2">
              <Code2 size={18} className="text-[var(--accent)]" />
              <span>DSA Patterns</span>
            </div>
            <div>Built with Next.js • Open Source</div>
          </div>
        </div>
      </footer>
    </div>
  );
}
