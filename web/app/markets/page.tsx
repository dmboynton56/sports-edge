import Link from "next/link";

import { MarketsTable } from "@/components/dashboard/MarketsTable";
import { PageHeader } from "@/components/dashboard/PageHeader";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { getProductionPredictionFeed } from "@/lib/data/player-markets";
import { SPORTS, type SportEntry } from "@/lib/markets-registry";
import { cn } from "@/lib/utils";

function SportCard({ sport, muted = false }: { sport: SportEntry; muted?: boolean }) {
  return (
    <Card className={cn(muted && "bg-muted/20 text-muted-foreground")}>
      <CardHeader className={cn(muted && "pb-2")}>
        <div className="flex items-center justify-between gap-3">
          <CardTitle className={cn(muted && "text-base")}>{sport.label}</CardTitle>
          {sport.emphasis !== "primary" ? (
            <Badge variant="outline" className="text-muted-foreground">
              {sport.emphasis}
            </Badge>
          ) : null}
        </div>
        <p className="text-sm text-muted-foreground">{sport.description}</p>
      </CardHeader>
      <CardContent className="space-y-2">
        {sport.markets.map((market) => (
          <Link
            key={market.slug}
            href={market.href}
            className="block rounded-lg border border-border bg-background/60 p-3 transition-colors hover:border-accent/50 hover:bg-accent/5"
          >
            <div className="flex flex-wrap items-center justify-between gap-2">
              <span className="text-sm font-medium text-foreground">{market.label}</span>
              {market.status === "scaffold" ? (
                <Badge variant="secondary" className="text-muted-foreground">
                  scaffold
                </Badge>
              ) : null}
            </div>
            <p className="mt-1 text-xs text-muted-foreground">{market.description}</p>
          </Link>
        ))}
      </CardContent>
    </Card>
  );
}

export default async function MarketsPage() {
  const feed = await getProductionPredictionFeed();
  const primarySports = SPORTS.filter((sport) => sport.emphasis === "primary");
  const secondarySports = SPORTS.filter((sport) => sport.emphasis !== "primary");

  return (
    <div>
      <PageHeader
        title="Markets"
        description="Choose a sport and market to inspect the numbers produced by each model."
        meta={feed.generatedAt}
      />

      <div className="grid gap-4 lg:grid-cols-3">
        {primarySports.map((sport) => (
          <SportCard key={sport.slug} sport={sport} />
        ))}
      </div>

      <div className="mt-4 grid gap-3 md:grid-cols-3">
        {secondarySports.map((sport) => (
          <SportCard key={sport.slug} sport={sport} muted />
        ))}
      </div>

      <Card className="mt-6">
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
