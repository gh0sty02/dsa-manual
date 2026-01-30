'use client';

import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark as darkTheme } from 'react-syntax-highlighter/dist/cjs/styles/prism';
import { oneLight as lightTheme } from 'react-syntax-highlighter/dist/cjs/styles/prism';
import { Check, Copy } from 'lucide-react';

interface MarkdownRendererProps {
  content: string;
}

interface CodeBlockProps {
  language: string;
  value: string;
}

function CodeBlock({ language, value }: CodeBlockProps) {
  const [copied, setCopied] = useState(false);

  const copyToClipboard = async () => {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  // Detect if user prefers dark mode
  const isDark =
    typeof window !== 'undefined'
      ? window.matchMedia('(prefers-color-scheme: dark)').matches
      : true;

  const displayLanguage = language || 'text';

  return (
    <div className="code-block-wrapper group">
      <div className="code-block-header">
        <span className="uppercase tracking-wider">{displayLanguage}</span>
        <button
          onClick={copyToClipboard}
          className={`copy-button ${copied ? 'copied' : ''}`}
          aria-label="Copy code"
        >
          {copied ? (
            <>
              <Check size={14} />
              Copied!
            </>
          ) : (
            <>
              <Copy size={14} />
              Copy
            </>
          )}
        </button>
      </div>
      <SyntaxHighlighter
        language={language || 'text'}
        style={isDark ? darkTheme : lightTheme}
        customStyle={{
          margin: 0,
          padding: '1rem 1.25rem',
          background: 'transparent',
          fontSize: '0.875rem',
          lineHeight: '1.7',
        }}
        codeTagProps={{
          style: {
            fontFamily: 'var(--font-mono)',
          },
        }}
      >
        {value}
      </SyntaxHighlighter>
    </div>
  );
}

export default function MarkdownRenderer({ content }: MarkdownRendererProps) {
  return (
    <article className="prose prose-lg max-w-none">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          // Custom code block rendering
          // Custom code block rendering
          code(props: any) {
            const { node, className, children, ...rest } = props;
            const match = /language-(\w+)/.exec(className || '');

            // React-markdown passes an 'inline' boolean prop.
            // If it's missing (shouldn't be in v8+), we can try to infer:
            // Block code usually ends with a newline, inline doesn't.
            const isInline = props.inline ?? !String(children).includes('\n');

            if (isInline) {
              return (
                <code className={className} {...rest}>
                  {children}
                </code>
              );
            }

            return (
              <CodeBlock
                language={match ? match[1] : 'text'}
                value={String(children).replace(/\n$/, '')}
              />
            );
          },
          // Remove the wrapping pre tag since CodeBlock handles it
          pre({ children }) {
            return <>{children}</>;
          },
          // Enhanced links
          a({ href, children, ...props }) {
            const isExternal = href?.startsWith('http');
            return (
              <a
                href={href}
                target={isExternal ? '_blank' : undefined}
                rel={isExternal ? 'noopener noreferrer' : undefined}
                {...props}
              >
                {children}
              </a>
            );
          },
          // Enhanced headings with anchor links
          h2({ children, ...props }) {
            const id = String(children)
              .toLowerCase()
              .replace(/\s+/g, '-')
              .replace(/[^\w-]/g, '');
            return (
              <h2 id={id} {...props}>
                {children}
              </h2>
            );
          },
          h3({ children, ...props }) {
            const id = String(children)
              .toLowerCase()
              .replace(/\s+/g, '-')
              .replace(/[^\w-]/g, '');
            return (
              <h3 id={id} {...props}>
                {children}
              </h3>
            );
          },
          // Enhanced tables
          table({ children, ...props }) {
            return (
              <div className="overflow-x-auto my-6 rounded-lg border border-[var(--border)]">
                <table {...props}>{children}</table>
              </div>
            );
          },
          // Enhanced blockquotes
          blockquote({ children, ...props }) {
            return <blockquote {...props}>{children}</blockquote>;
          },
        }}
      >
        {content}
      </ReactMarkdown>
    </article>
  );
}
