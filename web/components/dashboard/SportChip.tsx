import { sportColor } from "@/lib/sports";
import { cn } from "@/lib/utils";

/** A league label carrying its color. Used anywhere a sport is named in a list. */
export function SportChip({
  sport,
  label,
  muted = false,
  className,
}: {
  sport: string;
  label: string;
  muted?: boolean;
  className?: string;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-2 rounded-full bg-secondary px-2.5 py-1 text-xs font-semibold text-secondary-foreground",
        muted && "opacity-55",
        className,
      )}
    >
      <span className={cn("size-1.5 rounded-[2px]", sportColor(sport).fill)} />
      {label}
    </span>
  );
}

/** The taller swatch used to anchor a sport in a table row. */
export function SportSwatch({ sport, label }: { sport: string; label: string }) {
  return (
    <span className="flex items-center gap-2.5 font-display text-base font-bold tracking-tight text-foreground">
      <span className={cn("h-5 w-[3px] rounded-full", sportColor(sport).fill)} />
      {label}
    </span>
  );
}
