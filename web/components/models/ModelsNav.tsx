"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

import { cn } from "@/lib/utils";

const MODEL_NAV_ITEMS = [
  { href: "/models", label: "Overview" },
  { href: "/models/registry", label: "Registry" },
  { href: "/models/performance", label: "Performance" },
  { href: "/models/results", label: "Results" },
  { href: "/models/insights", label: "Insights" },
  { href: "/models/data-quality", label: "Data quality" },
];

export function ModelsNav() {
  const pathname = usePathname();

  return (
    <nav aria-label="Models" className="mb-7 border-b border-border">
      <div className="flex flex-wrap gap-1">
        {MODEL_NAV_ITEMS.map((item) => {
          const active = item.href === "/models"
            ? pathname === item.href
            : pathname.startsWith(item.href);
          return (
            <Link
              key={item.href}
              href={item.href}
              aria-current={active ? "page" : undefined}
              className={cn(
                "border-b-2 border-transparent px-3 py-2.5 text-sm font-medium text-muted-foreground transition-colors hover:text-foreground",
                active && "border-accent text-foreground",
              )}
            >
              {item.label}
            </Link>
          );
        })}
      </div>
    </nav>
  );
}
