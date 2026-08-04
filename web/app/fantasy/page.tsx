import { PageHeader } from "@/components/dashboard/PageHeader";
import { FantasyBoard } from "@/components/fantasy/FantasyBoard";
import { getFantasyFeed } from "@/lib/data/fantasy";

export default async function FantasyPage() {
  const feed = await getFantasyFeed("preseason");

  return (
    <div>
      <PageHeader
        title="Fantasy Football"
        description="Public NFL player projections, configurable full-PPR scoring, a live snake-draft board, and a local weekly lineup planner."
        meta={feed.generatedAt}
      />
      <FantasyBoard feed={feed} />
    </div>
  );
}
