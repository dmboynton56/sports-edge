import { cn } from "@/lib/utils";

export type StatTone = "default" | "up" | "down";

const TONE = {
  default: "",
  up: "text-positive",
  down: "text-destructive",
} satisfies Record<StatTone, string>;

/** A single figure with its label. Deliberately small — the page is navigation first. */
export function StatTile({
  label,
  value,
  suffix,
  tone = "default",
  className,
}: {
  label: string;
  value: string;
  suffix?: string;
  tone?: StatTone;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "rounded-lg border border-border bg-card px-4 py-3 shadow-soft",
        className,
      )}
    >
      <dt className="text-[11px] font-semibold text-muted-foreground">{label}</dt>
      <dd className={cn("figure mt-0.5 text-[28px] leading-tight", TONE[tone])}>
        {value}
        {suffix ? (
          <span className="ml-1 text-sm font-medium text-muted-foreground">{suffix}</span>
        ) : null}
      </dd>
    </div>
  );
}
