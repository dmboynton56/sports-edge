import Link from "next/link";

import { MarketsTable } from "@/components/dashboard/MarketsTable";
import { PageHeader } from "@/components/dashboard/PageHeader";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { getProductionPredictionFeed } from "@/lib/data/player-markets";
import { getSport } from "@/lib/markets-registry";

const WINNER_MARKETS = new Set(["winner", "moneyline", "money_line", "ml"]);

export default async function MlbMarketsPage() {
  const feed = await getProductionPredictionFeed();
  const mlb = getSport("mlb");
  const predictions = feed.predictions.filter(
    (prediction) =>
      prediction.sport.toLowerCase() === "mlb" &&
      WINNER_MARKETS.has(prediction.market.toLowerCase()),
  );

  return (
    <div>
      <PageHeader
        title="MLB Markets"
        description="Choose a baseball market or inspect live team winner probabilities."
        meta={feed.generatedAt}
      />

      <div className="mb-6 grid gap-3 sm:grid-cols-2">
        {mlb?.markets.map((market) => (
          <Link key={market.slug} href={market.href}>
            <Card className="h-full transition-colors hover:border-accent/50 hover:bg-accent/5">
              <CardHeader>
                <div className="flex items-center justify-between gap-2">
                  <CardTitle className="text-base">{market.label}</CardTitle>
                  {market.status === "scaffold" ? (
                    <Badge variant="secondary">scaffold</Badge>
                  ) : null}
                </div>
              </CardHeader>
              <CardContent className="text-sm text-muted-foreground">
                {market.description}
              </CardContent>
            </Card>
          </Link>
        ))}
      </div>

      <Card>
        <CardHeader>
          <CardTitle>MLB Winner Board</CardTitle>
        </CardHeader>
        <CardContent>
          {predictions.length > 0 ? (
            <MarketsTable initialPredictions={predictions} initialGaps={feed.gaps} />
          ) : (
            <div className="rounded-lg border border-dashed border-border bg-muted/20 px-4 py-10 text-center text-sm text-muted-foreground">
              No live MLB winner markets are available.
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
