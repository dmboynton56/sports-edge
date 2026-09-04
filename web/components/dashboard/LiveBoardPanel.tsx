import Link from "next/link";
import { ArrowRight } from "lucide-react";

import { SportChip } from "@/components/dashboard/SportChip";
import { isFiniteNumber } from "@/lib/data/json";
import type { Performance, Prediction } from "@/lib/data/types";
import { formatAmericanPrice, formatDateTime, formatNumber, formatPct } from "@/lib/format";
import { cn } from "@/lib/utils";

const STATUS_LABEL = {
  supported: "Supported",
  research: "Research",
  model_only: "Model only",
} satisfies Record<Prediction["marketStatus"], string>;

function Shell({
  eyebrow,
  note,
  action,
  children,
}: {
  eyebrow: string;
  note: string;
  action?: { label: string; href: string };
  children: React.ReactNode;
}) {
  return (
    // The one lifted surface on the page. Everything else sits on the ground,
    // so the eye lands here first.
    <section className="overflow-hidden rounded-2xl border border-border bg-card shadow-lift">
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1 border-b border-border px-5 py-4">
        <h2 className="text-[13px] font-semibold tracking-[0.06em] uppercase">{eyebrow}</h2>
        <p className="text-[13px] text-muted-foreground">{note}</p>
        {action ? (
          <Link
            href={action.href}
            className="ml-auto inline-flex items-center gap-1.5 text-[13px] font-semibold text-accent hover:underline"
          >
            {action.label}
            <ArrowRight className="size-3.5" />
          </Link>
        ) : null}
      </div>
      {children}
    </section>
  );
}

function BoardRow({ prediction }: { prediction: Prediction }) {
  const { edge } = prediction;
  const detail = [
    prediction.book === "model" ? "model only" : prediction.book,
    isFiniteNumber(prediction.line) ? `line ${formatNumber(prediction.line, 1)}` : null,
    formatAmericanPrice(prediction.price),
  ].filter(Boolean).join(" · ");

  const row = (
    <>
      <SportChip sport={prediction.sport.toLowerCase()} label={prediction.sport} className="shrink-0" />
      <div className="min-w-0">
        <div className="truncate text-[15px] font-medium">{prediction.subject}</div>
        <div className="truncate text-xs text-muted-foreground">
          {detail} · {STATUS_LABEL[prediction.marketStatus]}
        </div>
      </div>
      <div className="shrink-0 text-right">
        <div
          className={cn(
            "figure text-[19px] leading-tight",
            !isFiniteNumber(edge)
              ? "text-muted-foreground"
              : edge < 0
                ? "text-destructive"
                : "text-positive",
          )}
        >
          {isFiniteNumber(edge) ? formatPct(edge) : "—"}
        </div>
        <div className="text-[11px] text-muted-foreground">
          {isFiniteNumber(edge) ? "edge · " : "no price · "}
          model {formatPct(prediction.modelProbability, 0)}
        </div>
      </div>
    </>
  );

  const className =
    "flex items-center gap-3.5 border-b border-border px-5 py-3.5 last:border-b-0 sm:gap-5";

  return prediction.detailHref ? (
    <Link href={prediction.detailHref} className={cn(className, "transition-colors hover:bg-secondary")}>
      {row}
    </Link>
  ) : (
    <div className={className}>{row}</div>
  );
}

/**
 * The hero. Shows the live board when there is one, and stays useful when there
 * isn't: an empty slate falls back to the season record rather than a row of
 * zeros, because the board depends on feeds that go quiet overnight and between
 * seasons.
 */
export function LiveBoardPanel({
  predictions,
  generatedAt,
  records,
}: {
  predictions: Prediction[];
  generatedAt: string | null;
  records: Performance[];
}) {
  if (predictions.length > 0) {
    // A feed with no usable timestamp shouldn't advertise "updated n/a".
    const stamp = generatedAt ? formatDateTime(generatedAt) : "n/a";
    const note = `Top ${predictions.length} by expected value${stamp === "n/a" ? "" : ` · updated ${stamp}`}`;
    return (
      <Shell
        eyebrow="Live board"
        note={note}
        action={{ label: "Open the full board", href: "/markets" }}
      >
        {predictions.map((prediction) => (
          <BoardRow key={prediction.id} prediction={prediction} />
        ))}
      </Shell>
    );
  }

  const graded = records.filter((record) => (record.sampleSize ?? 0) > 0);
  if (graded.length > 0) {
    return (
      <Shell
        eyebrow="Season record"
        note="No games are open for betting right now, so here is the record so far."
        action={{ label: "See all performance", href: "/models/performance" }}
      >
        {graded.map((record) => (
          <div
            key={`${record.sport}-${record.modelVersion}-${record.market}`}
            className="flex items-center gap-3.5 border-b border-border px-5 py-3.5 last:border-b-0 sm:gap-5"
          >
            <SportChip sport={record.sport.toLowerCase()} label={record.sport} className="shrink-0" />
            <div className="min-w-0">
              <div className="truncate text-[15px] font-medium">{record.market}</div>
              <div className="truncate text-xs text-muted-foreground">
                {record.modelVersion} · {formatNumber(record.sampleSize)} graded
              </div>
            </div>
            <div className="shrink-0 text-right">
              <div
                className={cn(
                  "figure text-[19px] leading-tight",
                  isFiniteNumber(record.roi)
                    ? record.roi < 0
                      ? "text-destructive"
                      : "text-positive"
                    : "text-muted-foreground",
                )}
              >
                {isFiniteNumber(record.roi) ? formatPct(record.roi) : "—"}
              </div>
              <div className="text-[11px] text-muted-foreground">
                {isFiniteNumber(record.roi) ? "ROI" : "no odds history"}
              </div>
            </div>
          </div>
        ))}
      </Shell>
    );
  }

  return (
    <Shell eyebrow="Board" note="Nothing to show yet">
      <div className="px-5 py-7">
        <p className="max-w-[52ch] text-sm leading-relaxed text-muted-foreground">
          No upcoming markets and no graded history are loading right now. That is
          a data problem on our side, not an empty slate — the source status page
          says which feed is missing.
        </p>
        <Link
          href="/models/data-quality"
          className="mt-3 inline-flex items-center gap-1.5 text-[13px] font-semibold text-accent hover:underline"
        >
          Check source health
          <ArrowRight className="size-3.5" />
        </Link>
      </div>
    </Shell>
  );
}
