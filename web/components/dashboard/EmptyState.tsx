import type { ReactNode } from "react";

import { cn } from "@/lib/utils";

/**
 * An empty screen should say what's missing and what would fill it,
 * not just report that there's nothing here.
 */
export function EmptyState({
  title,
  description,
  detail,
  children,
  className,
}: {
  title: string;
  description: string;
  detail?: ReactNode;
  children?: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "grid min-h-72 place-items-center rounded-xl border border-dashed border-border bg-card/50 px-6 py-12 text-center",
        className,
      )}
    >
      <div className="max-w-md">
        <h3 className="font-display text-lg font-bold tracking-tight">{title}</h3>
        <p className="mt-2 text-sm leading-relaxed text-muted-foreground">{description}</p>
        {detail ? <div className="mt-4">{detail}</div> : null}
        {children ? <div className="mt-5">{children}</div> : null}
      </div>
    </div>
  );
}
