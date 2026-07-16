import { MarketsTable } from "@/components/dashboard/MarketsTable";
import { PageHeader } from "@/components/dashboard/PageHeader";
import { Card, CardContent } from "@/components/ui/card";
import { getProductionPredictionFeed } from "@/lib/data/player-markets";

export default async function NbaMarketsPage() {
  const feed = await getProductionPredictionFeed();
  const predictions = feed.predictions.filter(
    (prediction) => prediction.sport.toLowerCase() === "nba",
  );

  return (
    <div>
      <PageHeader
        title="NBA Markets"
        description="Pre-live NBA spread and winner probabilities from production models."
        meta={feed.generatedAt}
      />
      {predictions.length > 0 ? (
        <MarketsTable initialPredictions={predictions} initialGaps={feed.gaps} />
      ) : (
        <Card className="border-dashed bg-muted/20">
          <CardContent className="py-10 text-center text-sm text-muted-foreground">
            No live NBA markets — models resume in season.
          </CardContent>
        </Card>
      )}
    </div>
  );
}
