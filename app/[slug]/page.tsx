import { getAllPosts, getPostBySlug, getCategories } from '@/lib/api';
import MarkdownRenderer from '@/components/MarkdownRenderer';
import Sidebar from '@/components/Sidebar';
import MobileSidebar from '@/components/MobileSidebar';
import ScrollToTop from '@/components/ScrollToTop';
import TableOfContents from '@/components/TableOfContents';
import { notFound } from 'next/navigation';
import Link from 'next/link';
import {
  ArrowLeft,
  BookOpen,
  Code2,
  Github,
  Clock,
  BarChart,
  Tag,
  Layers,
} from 'lucide-react';

export async function generateStaticParams() {
  const posts = getAllPosts();
  return posts.map((post) => ({
    slug: post.slug,
  }));
}

export default async function PatternPage({
  params,
}: {
  params: Promise<{ slug: string }>;
}) {
  const { slug } = await params;
  const post = getPostBySlug(slug);
  const posts = getAllPosts();
  const categories = getCategories();

  if (!post) {
    notFound();
  }

  // Prepare sidebar items
  const sidebarItems = posts.map((p) => ({
    title: p.title,
    slug: p.slug,
    categories:
      p.categories && p.categories.length > 0 ? p.categories : ['Other'],
  }));

  // Estimate read time (fallback calculation)
  const wordCount = post.content.split(/\s+/).length;
  const calculatedReadTime = Math.ceil(wordCount / 200);
  const displayTime =
    post.estimatedReadingTime || `${calculatedReadTime} min read`;

  // Extract headings for TOC
  const headings = post.content
    .split('\n')
    .filter((line) => line.match(/^(#{2,3})\s+(.+)$/))
    .map((line) => {
      const match = line.match(/^(#{2,3})\s+(.+)$/);
      if (!match) return null;
      const level = match[1].length;
      // Strip emojis from text
      const emojiRegex =
        /(\u00a9|\u00ae|[\u2000-\u3300]|\ud83c[\ud000-\udfff]|\ud83d[\ud000-\udfff]|\ud83e[\ud000-\udfff])/g;
      const text = match[2].replace(emojiRegex, '').trim();

      const id = text
        .toLowerCase()
        .replace(/\s+/g, '-')
        .replace(/[^\w-]/g, '');
      return { id, text, level };
    })
    .filter(
      (h): h is { id: string; text: string; level: number } => h !== null,
    );

  return (
    <div className="min-h-screen bg-[var(--background)]">
      {/* Header */}
      <header className="sticky top-0 z-50 backdrop-blur-md bg-[var(--background)]/80 border-b border-[var(--border)]">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center">
              <MobileSidebar items={sidebarItems} categories={categories} />
              <Link href="/" className="flex items-center gap-3 group">
                <div className="p-2 rounded-lg bg-[var(--accent)] text-white group-hover:scale-105 transition-transform">
                  <Code2 size={20} />
                </div>
                <span className="font-display font-bold text-lg text-[var(--foreground)] hidden sm:block">
                  DSA Manual
                </span>
              </Link>
            </div>

            <div className="flex items-center gap-4">
              <Link
                href="/"
                className="flex items-center gap-2 text-sm font-medium text-[var(--foreground-muted)] hover:text-[var(--accent)] transition-colors"
                title="Back to Overview"
              >
                <ArrowLeft size={16} />
                <span className="hidden sm:inline">Overview</span>
              </Link>

              <div className="h-6 w-px bg-[var(--border)] hidden sm:block"></div>

              <Link
                href="https://github.com/gh0sty02/dsa-patterns"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 text-sm text-[var(--foreground-muted)] hover:text-[var(--foreground)] transition-colors"
              >
                <Github size={18} />
                <span className="hidden sm:inline">GitHub</span>
              </Link>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-[1440px] mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex gap-10 py-8 justify-center">
          {/* Navigation Sidebar (Left) */}
          <div className="hidden lg:block w-64 flex-shrink-0 order-1">
            <div className="sticky top-24 max-h-[calc(100vh-8rem)] overflow-y-auto pr-2 pb-10">
              <Sidebar items={sidebarItems} categories={categories} />
            </div>
          </div>

          {/* Main Content (Center) */}
          <main className="flex-1 min-w-0 pb-20 max-w-3xl order-2">
            <article>
              {/* Article Header */}
              <header className="mb-10 animate-fade-in">
                {/* Meta Badges */}
                <div className="flex flex-wrap items-center gap-3 mb-6">
                  {post.categories &&
                    post.categories.map((cat) => (
                      <span
                        key={cat}
                        className="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-semibold bg-[var(--accent-light)] text-[var(--accent)]"
                      >
                        <Tag size={12} className="mr-1.5" />
                        {cat}
                      </span>
                    ))}
                  {post.difficulty && (
                    <span
                      className={`
                      inline-flex items-center px-2.5 py-1 rounded-full text-xs font-semibold
                      ${post.difficulty === 'Beginner' ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400' : ''}
                      ${post.difficulty === 'Intermediate' ? 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400' : ''}
                      ${post.difficulty === 'Advanced' ? 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400' : ''}
                    `}
                    >
                      <BarChart size={12} className="mr-1.5" />
                      {post.difficulty}
                    </span>
                  )}
                  <span className="inline-flex items-center text-xs text-[var(--foreground-muted)] font-medium">
                    <Clock size={12} className="mr-1.5" />
                    {displayTime}
                  </span>
                </div>

                <h1 className="font-display text-4xl md:text-5xl font-extrabold text-[var(--foreground)] mb-6 leading-tight">
                  {post.title}
                </h1>

                {/* Prerequisites Box */}
                {post.prerequisites && post.prerequisites.length > 0 && (
                  <div className="flex items-start gap-3 p-4 rounded-lg bg-[var(--surface-hover)] border border-[var(--border)]">
                    <Layers
                      size={18}
                      className="text-[var(--foreground-muted)] mt-0.5 flex-shrink-0"
                    />
                    <div>
                      <span className="text-xs font-bold uppercase tracking-wider text-[var(--foreground-muted)] block mb-1">
                        Prerequisites
                      </span>
                      <div className="flex flex-wrap gap-2">
                        {post.prerequisites.map((prereq: string, i: number) => (
                          <span
                            key={i}
                            className="text-sm font-medium text-[var(--foreground)]"
                          >
                            {prereq}
                            {i < post.prerequisites!.length - 1 ? ',' : ''}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                )}
              </header>

              {/* Article Content */}
              <div className="animate-slide-up stagger-1">
                <MarkdownRenderer content={post.content} />
              </div>

              {/* Footer / Navigation */}
              <div className="mt-16 pt-8 border-t border-[var(--border)]">
                <div className="flex justify-between items-center text-[var(--foreground-muted)] text-sm">
                  <div>Learn patterns, not just problems.</div>
                  <ScrollToTop />
                </div>
              </div>
            </article>
          </main>

          {/* Table of Contents (Right) */}
          <div className="hidden xl:block w-64 flex-shrink-0 order-3">
            <div className="sticky top-24">
              <TableOfContents headings={headings} />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
