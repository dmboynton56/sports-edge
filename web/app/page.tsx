import { ChannelCard, type ChannelChip } from "@/components/dashboard/ChannelCard";
import { GapsBanner } from "@/components/dashboard/GapsBanner";
import { SectionHeading } from "@/components/dashboard/PageHeader";
import { SportSwatch } from "@/components/dashboard/SportChip";
import { StatTile } from "@/components/dashboard/StatTile";
import { Card } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { deriveDataQuality } from "@/lib/data/data-quality";
import { getMlbHomeRunBoardSnapshot } from "@/lib/data/player-markets";
import { getPerformanceHistory } from "@/lib/data/performance";
import { formatDateTime, formatNumber, formatPct, formatPctFromWhole } from "@/lib/format";

export const dynamic = "force-dynamic";

function listSentence(items: string[]) {
  if (items.length <= 1) return items[0] ?? "";
  return `${items.slice(0, -1).join(", ")} and ${items.at(-1)}`;
}

export default async function Home() {
  const [history, mlbHr] = await Promise.all([
    getPerformanceHistory(),
    getMlbHomeRunBoardSnapshot(),
  ]);
  const quality = deriveDataQuality(history);
  const unhealthySources = quality.filter((row) => row.status !== "ok").length + (mlbHr.status === "healthy" ? 0 : 1);
  const backtestRows = history.records.reduce((sum, row) => sum + (row.sampleSize ?? 0), 0);
  const noRoi = history.records
    .filter((row) => typeof row.roi !== "number")
    .map((row) => row.sport);
  const lowestCoverage = quality
    .filter((row) => typeof row.coveragePct === "number")
    .toSorted((a, b) => (a.coveragePct ?? 0) - (b.coveragePct ?? 0))[0];
  const gapSummary = [
    noRoi.length
      ? `${listSentence(noRoi)} ${noRoi.length > 1 ? "have" : "has"} no sportsbook odds history, so ROI is not reported on this page.`
      : null,
    lowestCoverage
      ? `Lowest upstream coverage is ${lowestCoverage.sport ?? lowestCoverage.source} at ${formatPctFromWhole(lowestCoverage.coveragePct)}.`
      : null,
    mlbHr.gaps.length ? `MLB HR board: ${mlbHr.gaps[0]}` : null,
  ].filter(Boolean).join(" ");
  const marketChips: ChannelChip[] = [{
    sport: "mlb",
    label: `MLB HR · ${mlbHr.status}`,
    muted: mlbHr.status !== "healthy" && mlbHr.status !== "partial",
  }];

  return (
    <div>
      <section className="grid items-end gap-10 pb-2 pt-6 lg:grid-cols-[minmax(0,1fr)_auto] lg:gap-12">
        <div>
          <span className="inline-flex items-center gap-2.5 rounded-full border border-border bg-card px-3.5 py-1.5 text-[13px] font-semibold text-secondary-foreground shadow-soft">
            <span className={`anim-live-pulse size-[7px] rounded-full ${mlbHr.status === "healthy" || mlbHr.status === "partial" ? "bg-positive" : "bg-warning"}`} />
            {formatNumber(mlbHr.counts.candidates)} MLB HR candidates · {mlbHr.status} · refreshed {formatDateTime(mlbHr.completedAt)}
          </span>

          <h1 className="mt-6 font-display text-[clamp(2.4rem,5.5vw,3.9rem)] font-bold leading-[1.02] tracking-[-0.028em]">
            Every pick,
            <span className="block text-accent">graded in public.</span>
          </h1>

          <p className="mt-5 max-w-[54ch] text-base leading-relaxed text-muted-foreground">
            A trusted vertical slice from model probability to sportsbook price and next-day grade. Start with today&apos;s board, then inspect the durable results history.
          </p>
        </div>

        <dl className="grid grid-cols-2 gap-2.5 lg:w-[300px]">
          <StatTile label="Current candidates" value={formatNumber(mlbHr.counts.candidates)} />
          <StatTile label="Priced candidates" value={formatNumber(mlbHr.counts.priced)} />
          <StatTile label="Top-25 coverage" value={formatPct(mlbHr.counts.top25Coverage)} />
          <StatTile label="Last refresh" value={formatDateTime(mlbHr.completedAt)} />
        </dl>
      </section>

      <SectionHeading title="Where to go" note="Trusted product surfaces" />

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-6">
        <ChannelCard
          className="lg:col-span-3"
          href="/markets"
          title="Markets"
          description="Open the trusted MLB HR board with all candidates, sportsbook coverage, and explicit model-only rows."
          chips={marketChips}
          cta="Open the board"
        />
        <ChannelCard
          className="lg:col-span-3"
          href="/results"
          title="Results"
          description="Official outcomes graded against the immutable pregame publication snapshot."
          figures={[{ value: formatNumber(backtestRows), label: "Backtest rows" }, { value: String(history.records.length), label: "Models" }]}
          cta="See the grades"
        />
        <ChannelCard
          className="lg:col-span-2"
          href="/performance"
          title="Performance"
          description="Persisted backtest evidence by sport and market, kept separate from live board health."
          cta="See the record"
        />
        <ChannelCard
          className="lg:col-span-2"
          href="/insights"
          title="Insights"
          description="Write-ups on what changed and what the models got wrong."
          figures={[{ value: "2", label: "Posts" }]}
          cta="Read the notes"
        />
        <ChannelCard
          className="lg:col-span-2"
          href="/data-quality"
          title="Data quality"
          description="Current board status, pricing coverage, and freshness for every upstream source."
          figures={[{ value: String(unhealthySources), label: "Need work", tone: unhealthySources ? "down" : "up" }, { value: String(quality.length + 1), label: "Sources" }]}
          cta="Check the sources"
        />
      </div>

      <SectionHeading title="Backtest record" note="Season to date" action={{ label: "All performance", href: "/performance" }} />

      <Card className="overflow-hidden">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Sport</TableHead>
              <TableHead className="hidden sm:table-cell">Model</TableHead>
              <TableHead>Market</TableHead>
              <TableHead className="text-right">Sample</TableHead>
              <TableHead className="text-right">ROI</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {history.records.map((record) => {
              const roi = record.roi;
              return (
                <TableRow key={`${record.sport}-${record.modelVersion}`}>
                  <TableCell><SportSwatch sport={record.sport} label={record.sport} /></TableCell>
                  <TableCell className="hidden sm:table-cell">{record.modelVersion}</TableCell>
                  <TableCell>{record.market}</TableCell>
                  <TableCell className="text-right">{formatNumber(record.sampleSize)}</TableCell>
                  <TableCell className={typeof roi === "number" ? `figure text-right text-[17px] ${roi < 0 ? "text-destructive" : "text-positive"}` : "text-right text-sm"}>
                    {typeof roi === "number" ? formatPct(roi) : "No odds history"}
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </Card>

      <GapsBanner count={history.gaps.length + mlbHr.gaps.length} summary={gapSummary || "All tracked sources are reporting."} />
    </div>
  );
}
