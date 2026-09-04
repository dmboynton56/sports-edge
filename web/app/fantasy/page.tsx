import { PageHeader } from "@/components/dashboard/PageHeader";
import { FantasyBoard } from "@/components/fantasy/FantasyBoard";
import { getFantasyFeed } from "@/lib/data/fantasy-server";
import type { FantasyFeed } from "@/lib/data/fantasy";

export default async function FantasyPage() {
  const feed = await getFantasyFeed("preseason");
  const boardFeed: FantasyFeed = { ...feed, weekly: {} };

  return (
    <div>
      <PageHeader
        title="Fantasy Football"
        description="Fresh NFL player projections, half-PPR draft recommendations, a local snake-draft session, and weekly lineup planning."
        meta={feed.generatedAt}
      />
      <FantasyBoard feed={boardFeed} />
    </div>
  );
}
