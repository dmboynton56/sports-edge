import { EmptyState } from "@/components/dashboard/EmptyState";
import { MarketsTable } from "@/components/dashboard/MarketsTable";
import { PageHeader } from "@/components/dashboard/PageHeader";
import { getProductionPredictionFeed } from "@/lib/data/player-markets";

export const dynamic = "force-dynamic";

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
        <EmptyState
          title="No NBA board right now"
          description="Spread and winner probabilities publish on game days during the season. The model's full record stays available in the meantime."
        />
      )}
    </div>
  );
}
