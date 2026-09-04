import Link from "next/link";
import { notFound } from "next/navigation";

import { FeatureDrivers } from "@/components/analysis/FeatureDrivers";
import { PageHeader } from "@/components/dashboard/PageHeader";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { getGameExplanation } from "@/lib/data/explanations";
import { getTeamSlateGame } from "@/lib/data/team-markets";
import { formatPct } from "@/lib/format";

export const dynamic = "force-dynamic";

function formatSpread(line: number | null) {
  if (line == null || !Number.isFinite(line)) return "n/a";
  return line > 0 ? `+${line.toFixed(1)}` : line.toFixed(1);
}

export default async function NflGamePage({ params }: { params: Promise<{ gameId: string }> }) {
  const { gameId } = await params;
  const [game, explanation] = await Promise.all([
    getTeamSlateGame("NFL", gameId),
    getGameExplanation(gameId, "NFL"),
  ]);

  if (!game) notFound();

  return (
    <div>
      <PageHeader
        title={`${game.awayTeam} @ ${game.homeTeam}`}
        description="NFL game detail with model spread, win probability, and feature drivers."
        meta={game.predictionTs ?? game.gameTimeUtc}
      />

      <div className="mb-4">
        <Link href="/markets/nfl" className="text-sm text-muted-foreground hover:underline">
          ← Back to NFL slate
        </Link>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Prediction</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 text-sm">
            {game.week != null ? (
              <div className="flex justify-between">
                <span>Week</span>
                <span>{game.week}</span>
              </div>
            ) : null}
            <div className="flex justify-between">
              <span>Book spread</span>
              <span className="font-mono">{formatSpread(game.bookSpread)}</span>
            </div>
            <div className="flex justify-between">
              <span>Model spread</span>
              <span className="font-mono">{formatSpread(game.modelSpread)}</span>
            </div>
            <div className="flex justify-between">
              <span>Edge</span>
              <span className="font-mono">{formatSpread(game.edgePts)}</span>
            </div>
            <div className="flex justify-between">
              <span>Home win prob</span>
              <span>{formatPct(game.homeWinProb)}</span>
            </div>
            <div className="flex flex-wrap gap-2 pt-2">
              <Badge variant={game.freshnessStatus === "fresh" ? "accent" : "missing"}>
                {game.freshnessStatus}
              </Badge>
              {explanation?.injuryAdjusted ? <Badge variant="outline">injury adjusted</Badge> : null}
              {game.injuryDataMissing ? <Badge variant="missing">injury data missing</Badge> : null}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Injury Context</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 text-sm">
            {explanation ? (
              <>
                <div className="flex justify-between">
                  <span>Home EPA delta</span>
                  <span>{explanation.homeInjuryDelta?.toFixed(2) ?? "0.00"}</span>
                </div>
                <div className="flex justify-between">
                  <span>Away EPA delta</span>
                  <span>{explanation.awayInjuryDelta?.toFixed(2) ?? "0.00"}</span>
                </div>
              </>
            ) : (
              <p className="text-muted-foreground">No persisted explanation row yet.</p>
            )}
          </CardContent>
        </Card>
      </div>

      <Card className="mt-4">
        <CardHeader>
          <CardTitle>Top Feature Drivers</CardTitle>
        </CardHeader>
        <CardContent>
          <FeatureDrivers features={explanation?.topFeatures ?? []} />
        </CardContent>
      </Card>
    </div>
  );
}
