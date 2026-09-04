import Link from "next/link";

/**
 * Known gaps stay visible but stop shouting: one line of plain language,
 * with the full list one click away.
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
  if (count === 0) {
    return (
      <div className="mt-4 flex flex-wrap items-center gap-4 rounded-xl border border-positive/25 bg-positive-soft px-5 py-4">
        <span className="text-xs font-bold text-positive">No open gaps</span>
        <span className="flex-1 text-sm text-positive">
          Every tracked source is reporting full coverage.
        </span>
      </div>
    );
  }

  return (
    <div className="mt-4 flex flex-wrap items-center gap-4 rounded-xl border border-destructive/20 bg-destructive-soft px-5 py-4">
      <span className="whitespace-nowrap text-xs font-bold text-destructive">
        {count} open {count === 1 ? "gap" : "gaps"}
      </span>
      <span className="min-w-[15rem] flex-1 text-sm text-destructive/90">{summary}</span>
      <Link
        href={href}
        className="text-[13px] font-semibold text-destructive hover:underline"
      >
        See what&apos;s missing →
      </Link>
    </div>
  );
}
