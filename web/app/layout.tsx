import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Sports Edge | Analytics & Top Picks",
  description: "Advanced predictive models for sports betting edges.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.className} min-h-screen bg-background text-foreground selection:bg-primary selection:text-primary-foreground`}>
        <div className="flex flex-col min-h-screen">
          <header className="sticky top-0 z-50 w-full border-b border-border/40 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
            <div className="container mx-auto flex h-16 items-center px-4">
              <div className="mr-4 flex">
                <a className="mr-6 flex items-center space-x-2" href="/">
                  <span className="hidden font-bold sm:inline-block text-xl tracking-tight">
                    SPORTS<span className="text-accent">EDGE</span>
                  </span>
                </a>
                <nav className="flex items-center space-x-6 text-sm font-medium">
                  <a className="transition-colors hover:text-foreground/80 text-foreground" href="/">Top Edges</a>
                  <a className="transition-colors hover:text-foreground/80 text-foreground/60" href="/nba">NBA</a>
                  <a className="transition-colors hover:text-foreground/80 text-foreground/60" href="/nfl">NFL</a>
                  <a className="transition-colors hover:text-foreground/80 text-foreground/60" href="/pga">PGA</a>
                  <a className="transition-colors hover:text-foreground/80 text-foreground/60" href="/cbb">CBB</a>
                </nav>
              </div>
            </div>
          </header>
          <main className="flex-1">
            {children}
          </main>
        </div>
      </body>
    </html>
  );
}
