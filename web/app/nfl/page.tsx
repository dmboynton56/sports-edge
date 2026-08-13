import { PageHeader } from "@/components/dashboard/PageHeader";
import { TeamSpreadBoard } from "@/components/leagues/TeamSpreadBoard";
import { getTeamSlateFeed } from "@/lib/data/team-markets";

export const dynamic = "force-dynamic";

export default async function NflPage() {
  const feed = await getTeamSlateFeed("NFL", { lookaheadDays: 7 });

  return (
    <div>
      <PageHeader
        title="NFL Spread Board"
        description="Current-week NFL model spreads, market lines, edges, and freshness from Supabase."
        meta={feed.generatedAt}
      />
      <TeamSpreadBoard feed={feed} detailBasePath="/nfl" />
    </div>
  );
}
