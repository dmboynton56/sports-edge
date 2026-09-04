import { EmptyState } from "@/components/dashboard/EmptyState";
import { MarketsTable } from "@/components/dashboard/MarketsTable";
import { PageHeader } from "@/components/dashboard/PageHeader";
import { getProductionPredictionFeed } from "@/lib/data/player-markets";

export const dynamic = "force-dynamic";

export default async function NflMarketsPage() {
  const feed = await getProductionPredictionFeed();
  const predictions = feed.predictions.filter(
    (prediction) => prediction.sport.toLowerCase() === "nfl",
  );
  const gaps = feed.gaps.filter((gap) => gap.toLowerCase().includes("nfl"));

  return (
    <div>
      <PageHeader
        title="NFL Markets"
        description="Week 1 moneyline, spread, total, and guarded anytime-touchdown markets. Team outputs remain preliminary; TD probabilities passed an out-of-time outcome holdout, with role and longshot filters applied before EV is shown."
        meta={feed.generatedAt}
      />
      {predictions.length > 0 ? (
        <MarketsTable initialPredictions={predictions} initialGaps={gaps} />
      ) : (
        <EmptyState
          title="No NFL board right now"
          description="NFL featured markets publish when a scheduled slate has both model predictions and sportsbook snapshots."
        />
      )}
    </div>
  );
}
