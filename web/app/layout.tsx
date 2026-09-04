import type { Metadata } from "next";
import { Inter } from "next/font/google";
import { AppShell } from "@/components/dashboard/AppShell";
import "./globals.css";

// One typeface for the whole product, the way SF Pro carries Apple's UI. The
// `opsz` axis is what makes that work: it does automatically what SF Pro Display
// vs. Text does by hand, refining the letterforms as type gets larger. Weight,
// size and tracking carry the hierarchy instead of a second family.
const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  axes: ["opsz"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "Sports Edge | Sports Analytics & Fantasy Football",
  description: "Sports betting model performance, NFL fantasy projections, and configurable lineup planning.",
};

// Applies the stored theme before first paint so the page never flashes the wrong one.
const themeScript = `
try {
  var stored = localStorage.getItem("sports-edge-theme");
  if (stored === "dark" || (!stored && matchMedia("(prefers-color-scheme: dark)").matches)) {
    document.documentElement.classList.add("dark");
  }
} catch (e) {}
`;

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    // The font variable must sit on <html>: Tailwind declares --font-sans on
    // :root, and a var() there can only see custom properties on the same element.
    <html lang="en" className={inter.variable} suppressHydrationWarning>
      <head>
        {/* Must stay a raw inline script: next/script defers even
            beforeInteractive behind its runtime queue, which paints first. */}
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
      </head>
      <body
        suppressHydrationWarning
        className="min-h-screen bg-background font-sans text-foreground selection:bg-accent selection:text-accent-foreground"
      >
        <AppShell>{children}</AppShell>
      </body>
    </html>
  );
}
