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
        description="NFL moneyline, spread, and total markets use sportsbook prices. Anytime-touchdown rows show guarded model probabilities and fair odds derived from those probabilities, with no sportsbook EV."
        meta={feed.generatedAt}
      />
      {predictions.length > 0 ? (
        <MarketsTable initialPredictions={predictions} initialGaps={gaps} />
      ) : (
        <EmptyState
          title="No NFL board right now"
          description="NFL team markets need a scheduled slate and sportsbook snapshots. Touchdown probabilities publish when the player model has a valid current slate."
        />
      )}
    </div>
  );
}
