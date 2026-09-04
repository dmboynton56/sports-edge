import Link from "next/link";
import { Suspense } from "react";

import { MarketsTable } from "@/components/dashboard/MarketsTable";
import { PageHeader, SectionHeading } from "@/components/dashboard/PageHeader";
import { Card } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { getUnifiedMarketFeed } from "@/lib/data/unified-markets";
import { formatDateTime } from "@/lib/format";
import { SPORTS } from "@/lib/markets-registry";

export const dynamic = "force-dynamic";

function MarketsTableFallback() {
  return (
    <div className="space-y-4">
      <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-5">
        {Array.from({ length: 5 }, (_, index) => <Skeleton key={index} className="h-9" />)}
      </div>
      <Skeleton className="h-12" />
      <Skeleton className="h-96" />
    </div>
  );
}

export default async function MarketsPage() {
  const feed = await getUnifiedMarketFeed();
  const drillDowns = SPORTS.flatMap((sport) => sport.markets
    .filter((market) => market.status === "live")
    .map((market) => ({ ...market, sport: sport.label })));

  return (
    <div>
      <PageHeader
        title="Markets"
        description="Every upcoming publication-eligible model prediction in one board, including supported prices, research signals, and model-only probabilities."
        meta={`Updated ${formatDateTime(feed.generatedAt)}`}
      />

      <Card className="overflow-hidden p-4 sm:p-5">
        <Suspense fallback={<MarketsTableFallback />}>
          <MarketsTable
            initialPredictions={feed.predictions}
            initialGaps={feed.warnings}
            emptyTitle="No upcoming markets"
            emptyDescription="No publication-eligible predictions are available in the current serving window. Open Warnings for feed and guardrail details."
          />
        </Suspense>
      </Card>

      <SectionHeading title="Drill-downs" note="Specialized boards and game views" />
      <nav aria-label="Market drill-downs" className="flex flex-wrap gap-2">
        {drillDowns.map((market) => (
          <Link
            key={`${market.sport}-${market.slug}`}
            href={market.href}
            className="rounded-full border border-border bg-card px-3 py-1.5 text-sm font-medium text-muted-foreground shadow-soft transition-colors hover:border-accent/40 hover:text-foreground"
          >
            {market.sport} · {market.short}
          </Link>
        ))}
      </nav>
    </div>
  );
}
