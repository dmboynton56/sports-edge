"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import type { ReactNode } from "react";
import { useState, useSyncExternalStore } from "react";
import { Menu, Moon, Sun } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import { TooltipProvider } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

const navItems = [
  { href: "/markets/mlb/home-runs", label: "Markets" },
  { href: "/models", label: "Models" },
  { href: "/fantasy", label: "Fantasy" },
  { href: "/record", label: "Record" },
];

function useActive() {
  const pathname = usePathname();
  return (href: string) => {
    if (href === "/record") {
      return pathname === "/record" || pathname.startsWith("/results") || pathname.startsWith("/performance");
    }
    if (href === "/markets/mlb/home-runs") {
      return pathname === "/" || pathname === "/markets" || pathname.startsWith("/markets/");
    }
    return pathname.startsWith(href);
  };
}

function BrandMark() {
  return (
    <span className="inline-block size-3 shrink-0 border border-current" />
  );
}

function DesktopNav() {
  const isActive = useActive();

  return (
    <nav className="ml-8 hidden items-center gap-6 lg:flex">
      {navItems.map((item) => (
        <Link
          key={item.href}
          href={item.href}
          aria-current={isActive(item.href) ? "page" : undefined}
          className={cn(
            "text-sm font-medium transition-colors",
            isActive(item.href) ? "text-foreground" : "text-muted-foreground hover:text-foreground",
          )}
        >
          {item.label}
        </Link>
      ))}
    </nav>
  );
}

function MobileNav({ onNavigate }: { onNavigate: () => void }) {
  const isActive = useActive();

  return (
    <nav className="flex flex-col gap-1">
      {navItems.map((item) => (
        <Link
          key={item.href}
          href={item.href}
          onClick={onNavigate}
          aria-current={isActive(item.href) ? "page" : undefined}
          className={cn(
            "px-3 py-2 text-sm font-medium transition-colors",
            isActive(item.href) ? "text-foreground" : "text-muted-foreground hover:text-foreground",
          )}
        >
          {item.label}
        </Link>
      ))}
    </nav>
  );
}

function subscribeToTheme(onChange: () => void) {
  const observer = new MutationObserver(onChange);
  observer.observe(document.documentElement, {
    attributes: true,
    attributeFilter: ["class"],
  });
  return () => observer.disconnect();
}

function ThemeToggle() {
  const dark = useSyncExternalStore(
    subscribeToTheme,
    () => document.documentElement.classList.contains("dark"),
    () => false,
  );

  function toggle() {
    const next = !dark;
    document.documentElement.classList.toggle("dark", next);
    window.localStorage.setItem("sports-edge-theme", next ? "dark" : "light");
  }

  return (
    <button
      type="button"
      onClick={toggle}
      aria-label={dark ? "Switch to light mode" : "Switch to dark mode"}
      className="text-muted-foreground transition-colors hover:text-foreground"
    >
      {dark ? <Sun className="size-4" /> : <Moon className="size-4" />}
    </button>
  );
}

export function AppShell({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(false);

  return (
    <TooltipProvider>
      <div className="min-h-screen bg-background text-foreground">
        <header className="border-b border-border bg-card">
          <div className="mx-auto flex h-14 max-w-[1200px] items-center gap-3 px-4 sm:px-6">
            <Link
              href="/"
              aria-label="Sports Edge home"
              className="flex items-center gap-1.5 text-sm font-medium"
            >
              <BrandMark />
              <span>Sports Edge</span>
            </Link>

            <DesktopNav />

            <div className="ml-auto flex items-center gap-4">
              <ThemeToggle />
              <Sheet open={open} onOpenChange={setOpen}>
                <SheetTrigger asChild>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="lg:hidden"
                    aria-label="Open navigation"
                  >
                    <Menu className="size-4" />
                  </Button>
                </SheetTrigger>
                <SheetContent side="right" className="w-64">
                  <SheetHeader>
                    <SheetTitle className="flex items-center gap-1.5 text-sm font-medium">
                      <BrandMark />
                      Sports Edge
                    </SheetTitle>
                  </SheetHeader>
                  <div className="mt-6">
                    <MobileNav onNavigate={() => setOpen(false)} />
                  </div>
                </SheetContent>
              </Sheet>
            </div>
          </div>
        </header>
        <main className="mx-auto w-full max-w-[1200px] px-4 pb-16 pt-6 sm:px-6">
          {children}
        </main>
        <footer className="border-t border-border bg-card">
          <div className="mx-auto w-full max-w-[1200px] px-4 py-4 text-xs text-muted-foreground sm:px-6">
            <Link href="/data-quality" className="hover:text-foreground">
              Data quality
            </Link>
          </div>
        </footer>
      </div>
    </TooltipProvider>
  );
}
