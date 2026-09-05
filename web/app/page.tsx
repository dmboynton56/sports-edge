import Link from "next/link";

import { GapsBanner } from "@/components/dashboard/GapsBanner";
import { HowItWorks } from "@/components/dashboard/HowItWorks";
import { LiveBoardPanel } from "@/components/dashboard/LiveBoardPanel";
import { SectionHeading } from "@/components/dashboard/PageHeader";
import { SportSwatch } from "@/components/dashboard/SportChip";
import { Button } from "@/components/ui/button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { isFiniteNumber } from "@/lib/data/json";
import { getPerformanceHistory } from "@/lib/data/performance";
import { getUnifiedMarketFeed } from "@/lib/data/unified-markets";
import { formatDateTime, formatNumber, formatPct } from "@/lib/format";

export const dynamic = "force-dynamic";

const BOARD_ROWS = 6;

function listSentence(items: string[]) {
  if (items.length <= 1) return items[0] ?? "";
  return `${items.slice(0, -1).join(", ")} and ${items.at(-1)}`;
}

export default async function Home() {
  const [history, feed] = await Promise.all([
    getPerformanceHistory(),
    getUnifiedMarketFeed(),
  ]);

  // The board is the hero, so it comes off the unified feed rather than any one
  // league's snapshot — a single cold source can no longer empty the page.
  const board = feed.predictions.slice(0, BOARD_ROWS);
  const live = board.length > 0;

  const rawStamp = feed.generatedAt ? formatDateTime(feed.generatedAt) : "n/a";
  const feedStamp = rawStamp === "n/a" ? null : rawStamp;

  const gradedRows = history.records.reduce((sum, row) => sum + (row.sampleSize ?? 0), 0);
  const season = history.records.find((row) => row.season)?.season;
  const noRoi = history.records.filter((row) => !isFiniteNumber(row.roi)).map((row) => row.sport);

  const gapSummary = [
    noRoi.length
      ? `${listSentence(noRoi)} ${noRoi.length > 1 ? "have" : "has"} no sportsbook odds history, so ROI is not reported.`
      : null,
    feed.warnings[0] ?? null,
  ].filter(Boolean).join(" ");

  return (
    <div className="flex flex-col gap-16 pb-4">
      <section className="grid items-start gap-9 pt-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.05fr)] lg:gap-12">
        <div className="lg:pt-4">
          <span className="inline-flex items-center gap-2.5 rounded-full border border-border bg-card px-3.5 py-1.5 text-[13px] font-semibold text-secondary-foreground shadow-soft">
            <span
              className={`anim-live-pulse size-[7px] rounded-full ${live ? "bg-positive" : "bg-warning"}`}
            />
            {live
              ? `${formatNumber(feed.predictions.length)} markets open${feedStamp ? ` · updated ${feedStamp}` : ""}`
              : "No games open for betting right now"}
          </span>

          <h1 className="mt-6 text-[clamp(2.4rem,5.5vw,3.9rem)] font-bold leading-[1.02] tracking-[-0.038em]">
            Model predictions.
            <span className="block">Sportsbook odds.</span>
            <span className="block text-accent">You see the edge.</span>
          </h1>

          <p className="mt-5 max-w-[52ch] text-base leading-relaxed text-muted-foreground">
            Model probabilities go up before the game starts, get compared against
            the sportsbook&apos;s number, and are settled the next day against what
            actually happened. Wins and losses both stay on the record.
          </p>

          <div className="mt-7 flex flex-wrap items-center gap-3">
            <Button asChild size="lg">
              <Link href="/markets">See today&apos;s board</Link>
            </Button>
            <Button asChild variant="outline" size="lg">
              <Link href="/models/performance">How it has done</Link>
            </Button>
          </div>
        </div>

        <LiveBoardPanel
          predictions={board}
          generatedAt={feed.generatedAt}
          records={history.records}
        />
      </section>

      {/* Hairline band, not tiles: the figures support the board, they don't compete with it. */}
      <section className="grid gap-6 border-y border-border py-7 sm:grid-cols-[repeat(3,auto)_minmax(0,1fr)] sm:items-center sm:gap-10">
        <div>
          <div className="figure text-[26px] leading-none">{formatNumber(gradedRows)}</div>
          <div className="mt-1.5 text-[11px] font-semibold tracking-[0.06em] uppercase text-muted-foreground">
            Rows graded
          </div>
        </div>
        <div>
          <div className="figure text-[26px] leading-none">{history.records.length}</div>
          <div className="mt-1.5 text-[11px] font-semibold tracking-[0.06em] uppercase text-muted-foreground">
            Leagues
          </div>
        </div>
        <div>
          <div className="figure text-[26px] leading-none">{season ?? "n/a"}</div>
          <div className="mt-1.5 text-[11px] font-semibold tracking-[0.06em] uppercase text-muted-foreground">
            Season
          </div>
        </div>
        <p className="max-w-[46ch] text-sm leading-relaxed text-muted-foreground">
          Nothing is removed after the fact. Where a model is losing money, the
          number below says so.
        </p>
      </section>

      {/* When the board has nothing to show it falls back to this same record,
          so rendering both would just repeat the table twice. */}
      {live ? (
      <section>
        <SectionHeading
          title="Track record"
          note="Season to date"
          action={{ label: "All performance", href: "/models/performance" }}
        />
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>League</TableHead>
              <TableHead className="hidden sm:table-cell">Model</TableHead>
              <TableHead>Market</TableHead>
              <TableHead className="text-right">Sample</TableHead>
              <TableHead className="text-right">ROI</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {history.records.map((record) => (
              <TableRow key={`${record.sport}-${record.modelVersion}`}>
                <TableCell><SportSwatch sport={record.sport} label={record.sport} /></TableCell>
                <TableCell className="hidden sm:table-cell">{record.modelVersion}</TableCell>
                <TableCell>{record.market}</TableCell>
                <TableCell className="text-right">{formatNumber(record.sampleSize)}</TableCell>
                <TableCell
                  className={
                    isFiniteNumber(record.roi)
                      ? `figure text-right text-[17px] ${record.roi < 0 ? "text-destructive" : "text-positive"}`
                      : "text-right text-sm text-muted-foreground"
                  }
                >
                  {isFiniteNumber(record.roi) ? formatPct(record.roi) : "No odds history"}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </section>
      ) : null}

      <section>
        <SectionHeading title="How it works" note="Probability, price, grade" />
        <HowItWorks />
      </section>

      <GapsBanner
        count={history.gaps.length + feed.warnings.length}
        summary={gapSummary || "All tracked sources are reporting."}
      />
    </div>
  );
}
