import { PageHeader } from "@/components/dashboard/PageHeader";
import { TeamSpreadBoard } from "@/components/leagues/TeamSpreadBoard";
import { getTeamSlateFeed } from "@/lib/data/team-markets";

export const dynamic = "force-dynamic";

export default async function NbaPage() {
  const feed = await getTeamSlateFeed("NBA", { lookaheadDays: 1 });

  return (
    <div>
      <PageHeader
        title="NBA Spread Board"
        description="Today's NBA model spreads, market lines, edges, and freshness from Supabase."
        meta={feed.generatedAt}
      />
      <TeamSpreadBoard feed={feed} detailBasePath="/nba" />
    </div>
  );
}
