import type { Metadata } from 'next';
import {
  Plus_Jakarta_Sans,
  Source_Sans_3,
  JetBrains_Mono,
} from 'next/font/google';
import './globals.css';

const plusJakarta = Plus_Jakarta_Sans({
  variable: '--font-display',
  subsets: ['latin'],
  display: 'swap',
  weight: ['400', '500', '600', '700', '800'],
});

const sourceSans = Source_Sans_3({
  variable: '--font-body',
  subsets: ['latin'],
  display: 'swap',
  weight: ['400', '500', '600'],
});

const jetbrainsMono = JetBrains_Mono({
  variable: '--font-mono',
  subsets: ['latin'],
  display: 'swap',
  weight: ['400', '500'],
});

export const metadata: Metadata = {
  title: 'DSA Patterns | Master Data Structures & Algorithms',
  description:
    'A comprehensive guide to mastering DSA patterns for coding interviews. Learn with visual explanations and practical examples.',
  keywords: [
    'DSA',
    'Data Structures',
    'Algorithms',
    'LeetCode',
    'Coding Interviews',
    'Patterns',
  ],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${plusJakarta.variable} ${sourceSans.variable} ${jetbrainsMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
