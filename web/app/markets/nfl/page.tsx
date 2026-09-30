import { EmptyState } from "@/components/dashboard/EmptyState";
import { MarketsTable } from "@/components/dashboard/MarketsTable";
import { PageHeader, SectionHeading } from "@/components/dashboard/PageHeader";
import { TeamSpreadBoard } from "@/components/leagues/TeamSpreadBoard";
import { getProductionPredictionFeed } from "@/lib/data/player-markets";
import { getTeamSlateFeed } from "@/lib/data/team-markets";

export const dynamic = "force-dynamic";

export default async function NflMarketsPage() {
  const [feed, slate] = await Promise.all([
    getProductionPredictionFeed(),
    getTeamSlateFeed("NFL"),
  ]);
  const predictions = feed.predictions.filter(
    (prediction) => prediction.sport.toLowerCase() === "nfl",
  );
  const gaps = feed.gaps.filter((gap) => gap.toLowerCase().includes("nfl"));

  return (
    <div>
      <PageHeader
        title="NFL Markets"
        description="This week's model win probabilities and projected spreads stay visible while sportsbook prices are unavailable. Priced markets and guarded touchdown probabilities appear below when available."
        meta={feed.generatedAt}
      />
      {slate.games.length > 0 ? (
        <TeamSpreadBoard feed={slate} detailBasePath="/markets/nfl" />
      ) : null}
      {predictions.length > 0 ? (
        <>
          <SectionHeading title="Market probabilities" note="Model only until a book price is captured" />
          <MarketsTable initialPredictions={predictions} initialGaps={gaps} />
        </>
      ) : slate.games.length === 0 ? (
        <EmptyState
          title="No NFL board right now"
          description="No NFL games or publishable model probabilities are available in the current week."
        />
      ) : null}
    </div>
  );
}
