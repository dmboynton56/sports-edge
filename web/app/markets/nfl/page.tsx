import { EmptyState } from "@/components/dashboard/EmptyState";
import { MarketsTable } from "@/components/dashboard/MarketsTable";
import { PageHeader } from "@/components/dashboard/PageHeader";
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
        <EmptyState
          title="No NFL board right now"
          description="Spread and winner probabilities resume when the season starts. Note that NFL odds coverage sits at 19.6%, so the backtest is thinner than the other leagues."
        />
      )}
    </div>
  );
}
