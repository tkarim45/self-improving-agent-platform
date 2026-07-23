import type { Metadata } from "next";
import Link from "next/link";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "self-improving-agent-platform",
  description:
    "DuckDB support agent with a closed evaluation-and-retraining loop — chat + admin console",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col">
        <header className="border-b border-neutral-200 dark:border-neutral-800">
          <nav className="mx-auto flex max-w-5xl items-center gap-6 px-6 py-3 text-sm">
            <span className="font-semibold tracking-tight">
              self-improving-agent-platform
            </span>
            <Link href="/" className="hover:underline">
              Chat
            </Link>
            <Link href="/dashboard" className="hover:underline">
              Dashboard
            </Link>
            <a
              href="https://github.com/tkarim45/self-improving-agent-platform"
              className="ml-auto text-neutral-500 hover:underline"
              target="_blank"
              rel="noreferrer"
            >
              GitHub
            </a>
          </nav>
        </header>
        <main className="mx-auto w-full max-w-5xl flex-1 px-6 py-8">{children}</main>
      </body>
    </html>
  );
}
