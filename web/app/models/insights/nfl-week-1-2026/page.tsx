import Link from "next/link";

import { Notice } from "@/components/dashboard/Notice";
import { PageHeader } from "@/components/dashboard/PageHeader";
import { Badge } from "@/components/ui/badge";
import { buttonVariants } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { isFiniteNumber } from "@/lib/data/json";
import {
  NFL_LIVE_MODEL_VERSION,
  NFL_WEEK1_SEASON,
  NFL_WEEK1_WEEK,
  buildNflWeekReview,
  type CoverResult,
  type NflWeekGame,
  type NflWeekRecord,
} from "@/lib/data/nfl-week-review";
import { getPerformanceHistory } from "@/lib/data/performance";
import { getGameResultRows } from "@/lib/data/results";
import { formatDate, formatNumber, formatPct } from "@/lib/format";
import { cn } from "@/lib/utils";

export const dynamic = "force-dynamic";

function resultVariant(result: CoverResult | null) {
  if (result === "win") return "accent" as const;
  if (result === "push" || result == null) return "outline" as const;
  return "missing" as const;
}

function metricNumber(value: string | number | null | undefined) {
  return isFiniteNumber(value) ? value : null;
}

function signed(value: number | null, digits = 1) {
  if (value == null || !Number.isFinite(value)) return "n/a";
  const formatted = value.toFixed(digits);
  return value > 0 ? `+${formatted}` : formatted;
}

function RecordLine({ label, record }: { label: string; record: NflWeekRecord }) {
  return (
    <div className="rounded-xl border border-border bg-card/60 px-4 py-3">
      <div className="text-xs uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className="mt-1 text-lg font-semibold">
        {record.wins}-{record.losses}-{record.pushes}
      </div>
      <div className="text-sm text-muted-foreground">
        {formatPct(record.hitRate)} hit · {record.units == null ? "n/a" : `${signed(record.units)}u`}
      </div>
    </div>
  );
}

function pickLabel(game: NflWeekGame) {
  if (!game.winnerPick) return "n/a";
  const team = game.winnerPick === "home" ? game.homeTeam : game.awayTeam;
  return `${team} ${formatPct(game.homeWinProb == null ? null : game.winnerPick === "home" ? game.homeWinProb : 1 - game.homeWinProb)}`;
}

function atsLabel(game: NflWeekGame) {
  if (!game.bookAtsSide || game.bookSpread == null) return "n/a";
  if (game.bookAtsSide === "pass") return "no edge";
  const team = game.bookAtsSide === "home" ? game.homeTeam : game.awayTeam;
  const line = game.bookAtsSide === "home" ? game.bookSpread : -game.bookSpread;
  return `${team} ${signed(line)}`;
}

export default async function NflWeek1InsightPage() {
  const [result, performance] = await Promise.all([
    getGameResultRows("NFL"),
    getPerformanceHistory(),
  ]);
  const live = buildNflWeekReview(result.rows);
  const v1 = buildNflWeekReview(result.rows, NFL_WEEK1_SEASON, NFL_WEEK1_WEEK, "v1");
  const liveHistory = performance.records.find(
    (record) => record.sport === "NFL" && record.modelVersion === NFL_LIVE_MODEL_VERSION,
  );

  return (
    <div>
      <PageHeader
        title="NFL Week 1 2026 recap"
        description="Graded live v2 against the published pregame snapshot. Sportsbook ATS is the model's side of the book number; model-spread cover is whether home beat our own number."
        meta={`${live.games.length} graded ${NFL_LIVE_MODEL_VERSION} games`}
      />

      <div className="mb-4 flex flex-wrap gap-3">
        <Link href="/markets/nfl" className={cn(buttonVariants({ variant: "outline", size: "sm" }))}>
          Week 2 board
        </Link>
        <Link href="/models/performance/nfl" className={cn(buttonVariants({ variant: "outline", size: "sm" }))}>
          NFL performance
        </Link>
      </div>

      <Notice
        className="mb-4"
        title="Week 1 is evidence, not a license to retune"
        items={[
          "n=16. Do not refit nfl-v2-live-20260906 on this week. The locked 2025 test is consumed; graded 2026 is the next untouched window.",
          "Live winner ranking was inverted (AUC 0.45). Huge Week 2 moneyline dogs are model-vs-book disagreements, not cleared edges.",
          "Daily Refresh skipped NFL odds today: The Odds API returned 401 quota exhausted, so the board is still on Sep 7–9 snapshots.",
        ]}
      />

      {result.gaps.length ? <Notice className="mb-4" title="Data caveats" items={result.gaps} /> : null}

      <div className="grid gap-3 md:grid-cols-3">
        <RecordLine label="Winner (v2 live)" record={live.winner} />
        <RecordLine label="Sportsbook ATS (v2 live)" record={live.bookAts} />
        <RecordLine label="Model-spread cover (v2 live)" record={live.modelSpread} />
      </div>

      <Card className="mt-4">
        <CardHeader>
          <CardTitle>What the week actually said</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3 text-sm leading-6 text-muted-foreground">
          <p>
            {NFL_LIVE_MODEL_VERSION} went {live.winner.wins}-{live.winner.losses} on winners
            ({formatPct(live.winner.hitRate)}) and {live.modelSpread.wins}-{live.modelSpread.losses}
            covering its own spread. Sportsbook ATS is {live.bookAts.wins}-{live.bookAts.losses}
            {live.bookAts.sample ? "" : " (no book number on the graded rows)"}.
            v1 was {v1.winner.wins}-{v1.winner.losses} SU and {v1.modelSpread.wins}-{v1.modelSpread.losses}
            vs its own number — better moneyline, worse spread.
          </p>
          <p>
            Mean home win probability {formatPct(live.avgHomeWinProb)} vs actual home wins{" "}
            {formatPct(live.actualHomeWinRate)} (bias {signed(live.homeProbabilityBias, 3)}).
            Spread MAE {formatNumber(live.spreadMae, 1)} pts. Persisted live Brier{" "}
            {formatNumber(liveHistory?.metrics.brier, 4)}, ECE{" "}
            {formatNumber(metricNumber(liveHistory?.metrics.ece_10 ?? liveHistory?.metrics.live_ece_10), 3)}.
            Injuries remain excluded from v2 until the point-in-time coverage gate passes.
          </p>
        </CardContent>
      </Card>

      <Card className="mt-4">
        <CardHeader>
          <CardTitle>Game log</CardTitle>
        </CardHeader>
        <CardContent>
          {live.games.length ? (
            <Table className="min-w-[960px] table-auto">
              <TableHeader>
                <TableRow>
                  <TableHead>Date</TableHead>
                  <TableHead>Score</TableHead>
                  <TableHead>Model ML</TableHead>
                  <TableHead>Winner</TableHead>
                  <TableHead>Model / book</TableHead>
                  <TableHead>Book ATS</TableHead>
                  <TableHead>Margin err</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {live.games.map((game) => (
                  <TableRow key={`${game.gameDate}-${game.awayTeam}-${game.homeTeam}`}>
                    <TableCell>{formatDate(game.gameDate)}</TableCell>
                    <TableCell>
                      {game.awayTeam} {game.awayScore} @ {game.homeTeam} {game.homeScore}
                    </TableCell>
                    <TableCell>{pickLabel(game)}</TableCell>
                    <TableCell>
                      <Badge variant={resultVariant(game.winnerResult)}>
                        {game.winnerResult ?? "n/a"}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-muted-foreground">
                      {signed(game.mySpread)} / {signed(game.bookSpread)}
                    </TableCell>
                    <TableCell>
                      <div className="flex items-center gap-2">
                        <Badge variant={resultVariant(game.bookAtsResult)}>
                          {game.bookAtsResult ?? "n/a"}
                        </Badge>
                        <span className="text-xs text-muted-foreground">{atsLabel(game)}</span>
                      </div>
                    </TableCell>
                    <TableCell>{formatNumber(game.marginError, 1)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          ) : (
            <p className="text-sm text-muted-foreground">No Week 1 v2 grades are in Supabase yet.</p>
          )}
        </CardContent>
      </Card>

      <Card className="mt-4">
        <CardHeader>
          <CardTitle>Week 2 changes that actually matter</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3 text-sm leading-6 text-muted-foreground">
          <p>
            <span className="font-medium text-foreground">Do not publish stale-book EV.</span>{" "}
            Daily Refresh 35129163342 skipped NFL odds (`401 Usage quota has been reached`) and
            CFB ESPN history (`400` on Aug 2024 scoreboard) then skipped book-spread repair.
            The live slate still showed Sep 7–9 DraftKings numbers — including a 9-point JAX/DEN
            disagreement and NYG +330 at ~50% — as if they were current.
          </p>
          <p>
            <span className="font-medium text-foreground">Fail closed on odds age.</span>{" "}
            Featured NFL rows now withhold edge/EV when the sportsbook snapshot is older than 48h.
            Model probabilities still show. That matches the product rule: missing or unusable
            prices → model-only, no invented EV.
          </p>
          <p>
            <span className="font-medium text-foreground">Keep Daily NFL work intact when CFB dies.</span>{" "}
            CFB refresh is continue-on-error; book-spread repair uses the same `always()` pattern
            as final-score sync. Thursday BUF–DET should not depend on an ESPN college archive call.
          </p>
          <p>
            <span className="font-medium text-foreground">Ops, not a new model.</span>{" "}
            Reserve Odds API credits for the Wednesday NFL preview, or wait for quota reset before
            treating the board as priced. Do not wire injuries into v2 this week. Do not blend the
            market residual track — it is still disabled until timestamp-safe coverage clears 90%.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
