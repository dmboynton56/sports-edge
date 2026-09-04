import Link from "next/link";

/**
 * Known gaps stay visible but stop shouting: a footnote rule rather than a
 * filled banner, so the board panel keeps the page's only real elevation. The
 * colour lives in a single dot and the link, not a full-bleed tint.
 */
export function GapsBanner({
  count,
  summary,
  href = "/models/data-quality",
}: {
  count: number;
  summary: string;
  href?: string;
}) {
  const clean = count === 0;

  return (
    <div className="flex flex-wrap items-baseline gap-x-3 gap-y-2 border-t border-border pt-5 text-sm">
      <span className="inline-flex items-center gap-2 whitespace-nowrap font-semibold">
        <span className={`size-1.5 shrink-0 rounded-full ${clean ? "bg-positive" : "bg-warning"}`} />
        {clean ? "No open gaps" : `${count} open ${count === 1 ? "gap" : "gaps"}`}
      </span>
      <span className="min-w-[15rem] flex-1 leading-relaxed text-muted-foreground">
        {clean ? "Every tracked source is reporting full coverage." : summary}
      </span>
      <Link href={href} className="font-semibold text-accent hover:underline">
        {clean ? "Source health" : "See what's missing"} →
      </Link>
    </div>
  );
}
