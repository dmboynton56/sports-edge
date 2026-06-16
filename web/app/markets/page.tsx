import { MarketsTable } from "@/components/dashboard/MarketsTable";
import { PageHeader } from "@/components/dashboard/PageHeader";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { getProductionPredictionFeed } from "@/lib/data/player-markets";
import Link from "next/link";

export default async function MarketsPage() {
  const feed = await getProductionPredictionFeed();

  return (
    <div>
      <PageHeader
        title="Markets"
        description="Pre-live model probabilities and market candidates across team, tournament, and player-prop surfaces."
        meta={feed.generatedAt}
      />
      <div className="mb-4 flex flex-wrap gap-2">
        <Button asChild variant="outline" size="sm">
          <Link href="/markets/mlb-home-runs">MLB HR Board</Link>
        </Button>
        <Button asChild variant="outline" size="sm">
          <Link href="/pga">PGA Tournament Board</Link>
        </Button>
      </div>
      <Card>
        <CardHeader>
          <CardTitle>Pre-Live Model Board</CardTitle>
        </CardHeader>
        <CardContent>
          <MarketsTable
            initialPredictions={feed.predictions}
            initialGaps={feed.gaps}
          />
        </CardContent>
      </Card>
    </div>
  );
}
