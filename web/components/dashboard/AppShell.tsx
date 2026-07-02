"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import type { ReactNode } from "react";
import { useEffect, useState } from "react";
import {
  Activity,
  BarChart3,
  DatabaseZap,
  LineChart,
  Menu,
  Moon,
  MoreHorizontal,
  Newspaper,
  Sun,
  Trophy,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

const navItems = [
  { href: "/", label: "Overview", icon: Activity },
  { href: "/markets", label: "Markets", icon: LineChart },
  { href: "/insights", label: "Insights", icon: Newspaper },
  { href: "/performance", label: "Performance", icon: BarChart3 },
  { href: "/data-quality", label: "Data Quality", icon: DatabaseZap },
  { href: "/pga", label: "PGA", icon: Trophy },
  { href: "/cbb", label: "CBB", icon: Trophy },
];

function NavLinks({ onNavigate }: { onNavigate?: () => void }) {
  const pathname = usePathname();

  return (
    <nav className="flex flex-col gap-1 lg:flex-row lg:items-center">
      {navItems.map((item) => {
        const Icon = item.icon;
        const active =
          item.href === "/" ? pathname === "/" : pathname.startsWith(item.href);
        return (
          <Button
            asChild
            key={item.href}
            variant={active ? "secondary" : "ghost"}
            size="sm"
            className={cn(
              "justify-start text-muted-foreground lg:justify-center",
              active && "text-foreground",
            )}
          >
            <Link href={item.href} onClick={onNavigate}>
              <Icon className="size-4" />
              {item.label}
            </Link>
          </Button>
        );
      })}
    </nav>
  );
}

function ThemeToggle() {
  const [dark, setDark] = useState(true);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    const stored = window.localStorage.getItem("sports-edge-theme");
    const prefersDark = stored ? stored === "dark" : true;
    setDark(prefersDark);
    document.documentElement.classList.toggle("dark", prefersDark);
    setMounted(true);
  }, []);

  useEffect(() => {
    if (!mounted) return;
    document.documentElement.classList.toggle("dark", dark);
  }, [dark, mounted]);

  function toggle() {
    const next = !dark;
    setDark(next);
    window.localStorage.setItem("sports-edge-theme", next ? "dark" : "light");
    document.documentElement.classList.toggle("dark", next);
  }

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button variant="outline" size="icon" onClick={toggle} aria-label="Toggle theme">
          {mounted && !dark ? <Sun /> : <Moon />}
        </Button>
      </TooltipTrigger>
      <TooltipContent>Switch black/white mode</TooltipContent>
    </Tooltip>
  );
}

export function AppShell({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(false);

  return (
    <TooltipProvider>
      <div className="min-h-screen bg-background text-foreground">
        <header className="sticky top-0 z-40 border-b border-border bg-background/90 backdrop-blur">
          <div className="mx-auto flex h-14 max-w-screen-2xl items-center gap-3 px-4 sm:px-6">
            <Sheet open={open} onOpenChange={setOpen}>
              <SheetTrigger asChild>
                <Button variant="outline" size="icon" className="lg:hidden" aria-label="Open navigation">
                  <Menu />
                </Button>
              </SheetTrigger>
              <SheetContent side="left" className="w-80">
                <SheetHeader>
                  <SheetTitle>SPORTS EDGE</SheetTitle>
                </SheetHeader>
                <div className="mt-6">
                  <NavLinks onNavigate={() => setOpen(false)} />
                </div>
              </SheetContent>
            </Sheet>

            <Link href="/" className="flex min-w-fit items-center gap-2 font-semibold">
              <span className="grid size-7 place-items-center rounded-md border border-accent/50 bg-accent/10 text-xs text-accent">
                SE
              </span>
              <span className="hidden sm:inline">SPORTS EDGE</span>
            </Link>

            <div className="hidden flex-1 justify-center lg:flex">
              <NavLinks />
            </div>

            <div className="ml-auto flex items-center gap-2">
              <ThemeToggle />
              <DropdownMenu>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <DropdownMenuTrigger asChild>
                      <Button variant="outline" size="icon" aria-label="Open dashboard menu">
                        <MoreHorizontal />
                      </Button>
                    </DropdownMenuTrigger>
                  </TooltipTrigger>
                  <TooltipContent>Dashboard actions</TooltipContent>
                </Tooltip>
                <DropdownMenuContent align="end" className="w-56">
                  <DropdownMenuLabel>Data Adapters</DropdownMenuLabel>
                  <DropdownMenuItem asChild>
                    <Link href="/data-quality">Review source coverage</Link>
                  </DropdownMenuItem>
                  <DropdownMenuItem asChild>
                    <Link href="/performance">Inspect model history</Link>
                  </DropdownMenuItem>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem asChild>
                    <Link href="/markets">Open market monitor</Link>
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
          </div>
        </header>
        <main className="mx-auto w-full max-w-screen-2xl px-4 py-5 sm:px-6 lg:py-6">
          {children}
        </main>
      </div>
    </TooltipProvider>
  );
}
