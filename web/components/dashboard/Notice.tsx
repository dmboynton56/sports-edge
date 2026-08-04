import { cn } from "@/lib/utils";

type Tone = "warning" | "error" | "info";

const TONE: Record<Tone, { box: string; head: string; body: string }> = {
  warning: {
    box: "border-warning/25 bg-warning-soft",
    head: "text-warning",
    body: "text-warning/90",
  },
  error: {
    box: "border-destructive/20 bg-destructive-soft",
    head: "text-destructive",
    body: "text-destructive/90",
  },
  info: {
    box: "border-border bg-secondary/60",
    head: "text-foreground",
    body: "text-muted-foreground",
  },
};

/**
 * Caveats that qualify the numbers next to them. One box, a short heading,
 * and the list — rather than a scatter of badges.
 */
export function Notice({
  title,
  items,
  tone = "warning",
  className,
}: {
  title: string;
  items: string[];
  tone?: Tone;
  className?: string;
}) {
  if (!items.length) return null;
  const styles = TONE[tone];

  return (
    <div className={cn("rounded-xl border px-4 py-3", styles.box, className)}>
      <div className={cn("text-xs font-bold", styles.head)}>{title}</div>
      <ul className="mt-2 space-y-1.5">
        {items.map((item) => (
          <li key={item} className={cn("flex gap-2 text-[13px] leading-relaxed", styles.body)}>
            <span aria-hidden="true">·</span>
            <span>{item}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
