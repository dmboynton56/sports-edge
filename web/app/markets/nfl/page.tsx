import { MarketsTable } from "@/components/dashboard/MarketsTable";
import { PageHeader } from "@/components/dashboard/PageHeader";
import { Card, CardContent } from "@/components/ui/card";
import { getProductionPredictionFeed } from "@/lib/data/player-markets";

export default async function NflMarketsPage() {
  const feed = await getProductionPredictionFeed();
  const predictions = feed.predictions.filter(
    (prediction) => prediction.sport.toLowerCase() === "nfl",
  );

  return (
    <div>
      <PageHeader
        title="NFL Markets"
        description="Seasonal NFL spread and winner probabilities from production models."
        meta={feed.generatedAt}
      />
      {predictions.length > 0 ? (
        <MarketsTable initialPredictions={predictions} initialGaps={feed.gaps} />
      ) : (
        <Card className="border-dashed bg-muted/20">
          <CardContent className="py-10 text-center text-sm text-muted-foreground">
            No live NFL markets — models resume in season.
          </CardContent>
        </Card>
      )}
    </div>
  );
}
