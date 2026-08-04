import Link from "next/link";
import { ArrowRight } from "lucide-react";

import { MarketsTable } from "@/components/dashboard/MarketsTable";
import { PageHeader, SectionHeading } from "@/components/dashboard/PageHeader";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { getProductionPredictionFeed } from "@/lib/data/player-markets";
import { SPORTS, type SportEntry } from "@/lib/markets-registry";
import { sportColor } from "@/lib/sports";

function SportCard({ sport }: { sport: SportEntry }) {
  const live = sport.markets.filter((market) => market.status === "live").length;

  return (
    <Card className="flex flex-col p-5">
      <div className="flex items-center gap-3">
        <span className={`h-6 w-[3px] rounded-full ${sportColor(sport.slug).fill}`} />
        <h3 className="font-display text-xl font-bold tracking-tight">{sport.label}</h3>
        <Badge variant={live ? "positive" : "outline"} className="ml-auto">
          {live ? `${live} live` : "No board yet"}
        </Badge>
      </div>
      <p className="mt-2 text-sm text-muted-foreground">{sport.description}</p>

      <div className="mt-4 flex flex-col gap-1.5">
        {sport.markets.map((market) => (
          <Link
            key={market.slug}
            href={market.href}
            className="group flex items-start gap-3 rounded-lg border border-border bg-background/60 px-3.5 py-3 transition-colors hover:border-accent/40 hover:bg-accent-soft"
          >
            <span className="min-w-0 flex-1">
              <span className="block text-sm font-semibold text-foreground">
                {market.label}
              </span>
              <span className="mt-0.5 block text-xs leading-relaxed text-muted-foreground">
                {market.description}
              </span>
            </span>
            <ArrowRight className="mt-0.5 size-4 shrink-0 text-muted-foreground transition-all group-hover:translate-x-0.5 group-hover:text-accent" />
          </Link>
        ))}
      </div>
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
        description="Pick a league to see today's model numbers next to the book's."
        meta={feed.generatedAt}
      />

      <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
        {primarySports.map((sport) => (
          <SportCard key={sport.slug} sport={sport} />
        ))}
      </div>

      <SectionHeading title="Not running yet" note="Scaffolded, waiting on models or season" />
      <div className="grid gap-3 md:grid-cols-3">
        {secondarySports.map((sport) => (
          <SportCard key={sport.slug} sport={sport} />
        ))}
      </div>

      <SectionHeading
        title="Pre-live model board"
        note={`${feed.predictions.length} lines across every live market`}
      />
      <Card className="overflow-hidden p-5">
        <MarketsTable initialPredictions={feed.predictions} initialGaps={feed.gaps} />
      </Card>
    </div>
  );
}
