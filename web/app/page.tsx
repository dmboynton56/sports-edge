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
import { getPerformanceHistory } from "@/lib/data/performance";
import { getProductionPredictionFeed } from "@/lib/data/player-markets";
import {
  formatDateTime,
  formatNumber,
  formatPct,
  formatPctFromWhole,
} from "@/lib/format";
import { SPORTS } from "@/lib/markets-registry";

function listSentence(items: string[]) {
  if (items.length <= 1) return items[0] ?? "";
  return `${items.slice(0, -1).join(", ")} and ${items.at(-1)}`;
}

export default async function Home() {
  const [history, predictionFeed] = await Promise.all([
    getPerformanceHistory(),
    getProductionPredictionFeed(),
  ]);
  const quality = deriveDataQuality(history);

  const linesToday = predictionFeed.predictions.length;
  const liveSports = SPORTS.filter((sport) =>
    sport.markets.some((market) => market.status === "live"),
  );
  const gradedTotal = history.records.reduce((sum, row) => sum + (row.sampleSize ?? 0), 0);

  // Sample-weighted, so a 285-game NFL read can't outweigh 1,175 NBA games.
  const scored = history.records.filter((row) => typeof row.roi === "number");
  const scoredSample = scored.reduce((sum, row) => sum + (row.sampleSize ?? 0), 0);
  const blendedRoi = scoredSample
    ? scored.reduce((sum, row) => sum + (row.roi ?? 0) * (row.sampleSize ?? 0), 0) / scoredSample
    : null;

  const unhealthySources = quality.filter((row) => row.status !== "ok").length;

  // Say what's blocking in plain language rather than dumping the raw gap strings.
  const noRoi = history.records
    .filter((row) => typeof row.roi !== "number")
    .map((row) => row.sport);
  const lowestCoverage = quality
    .filter((row) => typeof row.coveragePct === "number")
    .toSorted((a, b) => (a.coveragePct ?? 0) - (b.coveragePct ?? 0))[0];
  const gapSummary = [
    noRoi.length
      ? `${listSentence(noRoi)} ${noRoi.length > 1 ? "have" : "has"} no sportsbook odds history, so ${noRoi.length > 1 ? "they" : "it"} can't report ROI yet.`
      : null,
    lowestCoverage
      ? `Lowest coverage is ${lowestCoverage.sport ?? lowestCoverage.source} at ${formatPctFromWhole(lowestCoverage.coveragePct)}.`
      : null,
  ]
    .filter(Boolean)
    .join(" ");

  const marketChips: ChannelChip[] = SPORTS.flatMap((sport) =>
    sport.markets.map((market) => ({
      sport: sport.slug,
      label: `${sport.label} ${market.short}`,
      muted: market.status !== "live",
    })),
  ).toSorted((a, b) => Number(a.muted ?? false) - Number(b.muted ?? false));

  return (
    <div>
      <section className="grid items-end gap-10 pb-2 pt-6 lg:grid-cols-[minmax(0,1fr)_auto] lg:gap-12">
        <div>
          <span className="inline-flex items-center gap-2.5 rounded-full border border-border bg-card px-3.5 py-1.5 text-[13px] font-semibold text-secondary-foreground shadow-soft">
            <span className="anim-live-pulse size-[7px] rounded-full bg-positive" />
            {formatNumber(linesToday)} lines priced · updated{" "}
            {formatDateTime(predictionFeed.generatedAt)}
          </span>

          <h1 className="mt-6 font-display text-[clamp(2.4rem,5.5vw,3.9rem)] font-bold leading-[1.02] tracking-[-0.028em]">
            Every pick,
            <span className="block text-accent">graded in public.</span>
          </h1>

          <p className="mt-5 max-w-[54ch] text-base leading-relaxed text-muted-foreground">
            Models across six leagues, scored against what actually happened. Start
            wherever you want — today&apos;s numbers, the track record, or the draft
            board.
          </p>
        </div>

        <dl className="grid grid-cols-2 gap-2.5 lg:w-[300px]">
          <StatTile label="Lines today" value={formatNumber(linesToday)} />
          <StatTile
            label="Boards live"
            value={String(liveSports.length)}
            suffix={`/ ${SPORTS.length}`}
          />
          <StatTile label="Graded" value={formatNumber(gradedTotal)} />
          <StatTile
            label="Blended ROI"
            value={formatPct(blendedRoi)}
            tone={blendedRoi !== null && blendedRoi < 0 ? "down" : "up"}
          />
        </dl>
      </section>

      <SectionHeading title="Where to go" note="Five surfaces" />

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-6">
        <ChannelCard
          className="lg:col-span-3"
          href="/markets"
          title="Markets"
          description="Today's model number next to the book's number, for every league that has a live board."
          chips={marketChips}
          cta="Open the boards"
        />
        <ChannelCard
          className="lg:col-span-3"
          href="/performance"
          title="Performance"
          description="How each model version has actually done, by sport and market."
          figures={[
            {
              value: formatPct(blendedRoi),
              label: "Blended ROI",
              tone: blendedRoi !== null && blendedRoi < 0 ? "down" : "up",
            },
            { value: String(history.records.length), label: "Models" },
          ]}
          cta="See the record"
        />
        <ChannelCard
          className="lg:col-span-2"
          href="/fantasy"
          title="Fantasy"
          description="NFL projections with scoring you can tune, a live draft board, and a weekly planner."
          cta="Open the draft board"
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
          description="Coverage and freshness for every source feeding the models."
          figures={[
            {
              value: String(unhealthySources),
              label: "Need work",
              tone: unhealthySources ? "down" : "up",
            },
            { value: String(quality.length), label: "Sources" },
          ]}
          cta="Check the sources"
        />
      </div>

      <SectionHeading
        title="The record"
        note="Season to date"
        action={{ label: "All performance", href: "/performance" }}
      />

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
                  <TableCell>
                    <SportSwatch sport={record.sport} label={record.sport} />
                  </TableCell>
                  <TableCell className="hidden sm:table-cell">
                    {record.modelVersion}
                  </TableCell>
                  <TableCell>{record.market}</TableCell>
                  <TableCell className="text-right">
                    {formatNumber(record.sampleSize)}
                  </TableCell>
                  <TableCell
                    className={
                      typeof roi === "number"
                        ? `figure text-right text-[17px] ${roi < 0 ? "text-destructive" : "text-positive"}`
                        : "text-right text-sm"
                    }
                  >
                    {typeof roi === "number" ? formatPct(roi) : "No odds history"}
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </Card>

      <GapsBanner
        count={history.gaps.length}
        summary={gapSummary || "All tracked sources are reporting."}
      />
    </div>
  );
}
