import { MarketsTable } from "@/components/dashboard/MarketsTable";
import { PageHeader } from "@/components/dashboard/PageHeader";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { getLocalPredictions } from "@/lib/data/predictions";

export default async function MarketsPage() {
  const feed = await getLocalPredictions();

  return (
    <div>
      <PageHeader
        title="Markets"
        description="Pre-live prediction monitor for book lines, implied probabilities, model edges, EV, Kelly sizing, confidence, and model versions."
        meta={feed.generatedAt}
      />
      <Card>
        <CardHeader>
          <CardTitle>Pre-Live Edges</CardTitle>
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
