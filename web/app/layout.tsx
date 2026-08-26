import type { Metadata } from "next";
import { IBM_Plex_Sans, Newsreader } from "next/font/google";
import { AppShell } from "@/components/dashboard/AppShell";
import "./globals.css";

const sans = IBM_Plex_Sans({
  weight: ["400", "500", "600", "700"],
  subsets: ["latin"],
  variable: "--font-sans",
  display: "swap",
});

const display = Newsreader({
  weight: ["400", "600", "700"],
  subsets: ["latin"],
  variable: "--font-display",
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
  if (stored === "dark") {
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
    // The font variables must sit on <html>: Tailwind declares --font-sans on
    // :root, and a var() there can only see custom properties on the same element.
    <html
      lang="en"
      className={`${sans.variable} ${display.variable}`}
      suppressHydrationWarning
    >
      <head>
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
