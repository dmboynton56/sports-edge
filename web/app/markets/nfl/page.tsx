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
        description="Moneyline, spread, total, and guarded anytime-touchdown markets. Team outputs are a monitored v2 rollout; stale sportsbook snapshots stay model-only with no EV."
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
