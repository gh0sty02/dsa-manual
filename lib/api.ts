import fs from 'fs';
import path from 'path';
import matter from 'gray-matter';

// The markdown files are in the content directory
const postsDirectory = path.join(process.cwd(), 'content');

export interface Post {
  slug: string;
  title: string;
  content: string;
  category?: string;
  difficulty?: string;
  prerequisites?: string[];
  estimatedReadingTime?: string;
  [key: string]: any;
}

export interface Category {
  name: string;
  slug: string;
  count: number;
}

function extractMetadata(content: string) {
  let cleanContent = content;
  let difficulty = 'Intermediate';
  let prerequisites: string[] = [];
  let estimatedReadingTime = '';

  // Extract Difficulty
  const diffMatch = content.match(/\*\*Difficulty:\*\*\s*(.+?)(?:\n|$)/i);
  if (diffMatch) {
    const diffText = diffMatch[1].toLowerCase();
    if (diffText.includes('beginner') || diffText.includes('easy'))
      difficulty = 'Beginner';
    else if (diffText.includes('advanced') || diffText.includes('hard'))
      difficulty = 'Advanced';
    else difficulty = 'Intermediate';

    cleanContent = cleanContent.replace(diffMatch[0], '');
  }

  // Extract Prerequisites
  const prereqMatch = content.match(/\*\*Prerequisites:\*\*\s*(.+?)(?:\n|$)/i);
  if (prereqMatch) {
    prerequisites = prereqMatch[1].split(',').map((s) => s.trim());
    cleanContent = cleanContent.replace(prereqMatch[0], '');
  }

  // Extract Reading Time
  const timeMatch = content.match(
    /\*\*Estimated Reading Time:\*\*\s*(.+?)(?:\n|$)/i,
  );
  if (timeMatch) {
    estimatedReadingTime = timeMatch[1].trim();
    cleanContent = cleanContent.replace(timeMatch[0], '');
  }

  // Clean up extra newlines at start
  cleanContent = cleanContent.replace(/^\s+/, '');

  return { cleanContent, difficulty, prerequisites, estimatedReadingTime };
}

// Pattern category mapping based on filename prefixes/keywords
const categoryKeywords: Record<string, string[]> = {
  'Dynamic Programming': ['knapsack', 'fibonacci', 'dynamic-programming'],
  'Two Pointers & Sliding Window': [
    'two-pointers',
    'sliding-window',
    'fast-slow',
  ],
  'Arrays & Sorting': ['cyclic-sort', 'merge-intervals'],
  'Linked Lists': ['linked-list', 'reversal'],
  'Stacks & Queues': ['stacks', 'monotonic-stack'],
  'Hash Maps & Sets': ['hash-maps', 'ordered-set'],
  'Trees & Graphs': [
    'tree-bfs',
    'graphs',
    'island',
    'trie',
    'topological',
    'union-find',
  ],
  Heaps: ['two-heaps', 'top-k', 'k-way-merge'],
  'Recursion & Backtracking': ['subsets', 'backtracking'],
  'Binary Search': ['binary-search'],
  'Bit Manipulation': ['bitwise', 'xor'],
  Greedy: ['greedy'],
  'Math & Prefix': ['prefix-sum'],
};

function getCategoryFromSlug(slug: string): string {
  const lowerSlug = slug.toLowerCase();

  for (const [category, keywords] of Object.entries(categoryKeywords)) {
    if (keywords.some((keyword) => lowerSlug.includes(keyword))) {
      return category;
    }
  }

  return 'Other';
}

// Remove old getDifficultyFromContent

export function getPostSlugs() {
  try {
    const fileNames = fs.readdirSync(postsDirectory);
    return fileNames.filter(
      (fileName) =>
        fileName.endsWith('.md') &&
        fileName !== 'README.md' &&
        fileName !== 'index.md',
    );
  } catch (e) {
    console.error('Error reading directory:', e);
    return [];
  }
}

export function getPostBySlug(slug: string): Post | null {
  try {
    const realSlug = slug.replace(/\.md$/, '');
    const fullPath = path.join(postsDirectory, `${realSlug}.md`);

    if (!fs.existsSync(fullPath)) {
      return null;
    }

    const fileContents = fs.readFileSync(fullPath, 'utf8');
    const { data, content } = matter(fileContents);

    // If title isn't in frontmatter, try to find h1 or use slug
    let title = data.title;
    if (!title) {
      const h1Match = content.match(/^#\s+(.+)$/m);
      if (h1Match) {
        title = h1Match[1];
      } else {
        // Clean up slug to make a readable title
        title = realSlug
          .replace(/^\d+-/, '') // Remove leading number
          .replace(/-/g, ' ')
          .replace(/\b\w/g, (l) => l.toUpperCase()); // Title case
      }
    }

    // Extract metadata and clean content
    const { cleanContent, difficulty, prerequisites, estimatedReadingTime } =
      extractMetadata(content);
    const category = getCategoryFromSlug(realSlug);

    return {
      slug: realSlug,
      title,
      content: cleanContent,
      category,
      difficulty,
      prerequisites,
      estimatedReadingTime,
      ...data,
    };
  } catch (e) {
    console.error(`Error reading post ${slug}:`, e);
    return null;
  }
}

export function getAllPosts(): Post[] {
  const slugs = getPostSlugs();
  const posts = slugs
    .map((slug) => getPostBySlug(slug))
    .filter((post): post is Post => post !== null)
    .sort((post1, post2) => (post1.slug > post2.slug ? 1 : -1));
  return posts;
}

export function getCategories(): Category[] {
  const posts = getAllPosts();
  const categoryMap = new Map<string, number>();

  posts.forEach((post) => {
    const category = post.category || 'Other';
    categoryMap.set(category, (categoryMap.get(category) || 0) + 1);
  });

  return Array.from(categoryMap.entries())
    .map(([name, count]) => ({
      name,
      slug: name.toLowerCase().replace(/\s+/g, '-').replace(/&/g, 'and'),
      count,
    }))
    .sort((a, b) => b.count - a.count);
}

export function getPostsByCategory(categorySlug: string): Post[] {
  const posts = getAllPosts();
  return posts.filter((post) => {
    const postCategorySlug = (post.category || 'Other')
      .toLowerCase()
      .replace(/\s+/g, '-')
      .replace(/&/g, 'and');
    return postCategorySlug === categorySlug;
  });
}
