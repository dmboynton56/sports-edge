"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import type { ReactNode } from "react";
import { useState, useSyncExternalStore } from "react";
import { Menu, Moon, Sun } from "lucide-react";

import { BrandMark } from "@/components/dashboard/BrandMark";
import { Button } from "@/components/ui/button";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import { TooltipProvider } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

const navItems = [
  { href: "/", label: "Overview" },
  { href: "/markets", label: "Markets" },
  { href: "/models", label: "Models" },
  { href: "/fantasy", label: "Fantasy" },
];

function useActive() {
  const pathname = usePathname();
  return (href: string) =>
    href === "/" ? pathname === "/" : pathname.startsWith(href);
}

/** Desktop nav: pills riding in a sunken track. */
function NavTrack() {
  const isActive = useActive();

  return (
    <nav className="ml-auto hidden rounded-xl bg-secondary p-1 lg:flex">
      {navItems.map((item) => (
        <Link
          key={item.href}
          href={item.href}
          aria-current={isActive(item.href) ? "page" : undefined}
          className={cn(
            "rounded-lg px-3.5 py-1.5 text-sm font-medium text-muted-foreground transition-colors hover:text-foreground",
            isActive(item.href) &&
              "bg-card font-semibold text-foreground shadow-soft hover:text-foreground",
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
            "rounded-lg px-3 py-2.5 text-base font-medium text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground",
            isActive(item.href) && "bg-secondary font-semibold text-foreground",
          )}
        >
          {item.label}
        </Link>
      ))}
    </nav>
  );
}

// The theme lives on <html> (set by the pre-paint script in layout.tsx), so the
// class list is the source of truth and React subscribes to it rather than
// keeping a second copy.
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
    <Button
      variant="outline"
      size="icon"
      onClick={toggle}
      aria-label={dark ? "Switch to light mode" : "Switch to dark mode"}
    >
      {dark ? <Sun /> : <Moon />}
    </Button>
  );
}

export function AppShell({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(false);

  return (
    <TooltipProvider>
      <div className="min-h-screen bg-background text-foreground">
        <header className="sticky top-0 z-40 border-b border-border bg-background/85 backdrop-blur-md">
          <div className="mx-auto flex h-16 max-w-[1200px] items-center gap-3 px-4 sm:px-7">
            <Link
              href="/"
              aria-label="Sports Edge home"
              className="flex min-w-fit items-center gap-2.5 font-display text-lg font-semibold tracking-[-0.03em]"
            >
              <BrandMark />
              <span>Sports Edge</span>
            </Link>

            <NavTrack />

            <div className="ml-auto flex items-center gap-2 lg:ml-2">
              <ThemeToggle />
              <Sheet open={open} onOpenChange={setOpen}>
                <SheetTrigger asChild>
                  <Button
                    variant="outline"
                    size="icon"
                    className="lg:hidden"
                    aria-label="Open navigation"
                  >
                    <Menu />
                  </Button>
                </SheetTrigger>
                <SheetContent side="right" className="w-72">
                  <SheetHeader>
                    <SheetTitle className="flex items-center gap-2.5 font-display text-lg font-semibold tracking-[-0.03em]">
                      <BrandMark />
                      Sports Edge
                    </SheetTitle>
                    <SheetDescription className="sr-only">
                      Primary Sports Edge navigation
                    </SheetDescription>
                  </SheetHeader>
                  <div className="mt-8">
                    <MobileNav onNavigate={() => setOpen(false)} />
                  </div>
                </SheetContent>
              </Sheet>
            </div>
          </div>
        </header>
        <main className="mx-auto w-full max-w-[1200px] px-4 pb-20 pt-7 sm:px-7">
          {children}
        </main>
      </div>
    </TooltipProvider>
  );
}
