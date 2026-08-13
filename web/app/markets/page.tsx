import Link from "next/link";
import { ArrowRight, ShieldCheck } from "lucide-react";

import { PageHeader, SectionHeading } from "@/components/dashboard/PageHeader";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { getMlbHomeRunBoardSnapshot } from "@/lib/data/player-markets";
import { SPORTS, type MarketEntry, type SportEntry } from "@/lib/markets-registry";
import { formatDateTime, formatNumber, formatPct } from "@/lib/format";

export const dynamic = "force-dynamic";

function statusLabel(sport: SportEntry, market: MarketEntry) {
  if (sport.slug === "mlb" && market.slug === "home-runs") return "Trusted slice";
  if (sport.emphasis === "seasonal") return "Seasonal";
  return market.status === "live" ? "Research" : "Candidate";
}

function MarketCard({ sport, market, status }: { sport: SportEntry; market: MarketEntry; status: string }) {
  return (
    <Link href={market.href} className="group block">
      <Card className="h-full transition-colors group-hover:border-accent/50">
        <CardHeader className="flex flex-row items-start justify-between gap-3">
          <div>
            <CardTitle className="text-base">{sport.label} · {market.label}</CardTitle>
            <p className="mt-2 text-sm leading-relaxed text-muted-foreground">{market.description}</p>
          </div>
          <Badge variant={status === "Trusted slice" ? "positive" : "outline"}>{status}</Badge>
        </CardHeader>
        <CardContent className="flex items-center justify-between pt-0 text-sm font-medium text-accent">
          Open market <ArrowRight className="size-4 transition-transform group-hover:translate-x-0.5" />
        </CardContent>
      </Card>
    </Link>
  );
}

export default async function MarketsPage() {
  const snapshot = await getMlbHomeRunBoardSnapshot();
  const mlb = SPORTS.find((sport) => sport.slug === "mlb");
  const trustedMarket = mlb?.markets.find((market) => market.slug === "home-runs");
  const otherMarkets = SPORTS.flatMap((sport) => sport.markets
    .filter((market) => !(sport.slug === "mlb" && market.slug === "home-runs"))
    .map((market) => ({ sport, market })));

  return (
    <div>
      <PageHeader
        title="Markets"
        description="One trusted MLB HR path is live end to end. Other markets are clearly labeled until their serving and grading contracts are complete."
        meta={`MLB HR ${snapshot.status} · refreshed ${formatDateTime(snapshot.completedAt)}`}
      />

      {trustedMarket && mlb ? (
        <Link href={trustedMarket.href} className="group block">
          <Card className="border-accent/30 bg-accent-soft/30 transition-colors group-hover:border-accent/60">
            <CardHeader className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
              <div className="flex gap-3">
                <ShieldCheck className="mt-0.5 size-5 text-accent" />
                <div>
                  <CardTitle>Trusted MLB HR board</CardTitle>
                  <p className="mt-2 max-w-2xl text-sm leading-relaxed text-muted-foreground">All current candidates, immutable sportsbook snapshots, top-25 pricing coverage, and explicit model-only labels.</p>
                </div>
              </div>
              <Badge variant={snapshot.status === "healthy" ? "positive" : snapshot.status === "partial" ? "warning" : "missing"}>{snapshot.status}</Badge>
            </CardHeader>
            <CardContent className="grid gap-3 pt-0 text-sm sm:grid-cols-4">
              <div><div className="text-xs text-muted-foreground">Candidates</div><div className="font-semibold">{formatNumber(snapshot.counts.candidates)}</div></div>
              <div><div className="text-xs text-muted-foreground">Priced</div><div className="font-semibold">{formatNumber(snapshot.counts.priced)}</div></div>
              <div><div className="text-xs text-muted-foreground">Top-25 coverage</div><div className="font-semibold">{formatPct(snapshot.counts.top25Coverage)}</div></div>
              <div><div className="text-xs text-muted-foreground">Last refresh</div><div className="font-semibold">{formatDateTime(snapshot.completedAt)}</div></div>
            </CardContent>
          </Card>
        </Link>
      ) : null}

      <SectionHeading title="Other markets" note="Serving status is explicit" />
      <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
        {otherMarkets.map(({ sport, market }) => (
          <MarketCard key={`${sport.slug}-${market.slug}`} sport={sport} market={market} status={statusLabel(sport, market)} />
        ))}
      </div>
    </div>
  );
}
