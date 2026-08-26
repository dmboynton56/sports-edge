import Link from "next/link";
import { PageHeader } from "@/components/dashboard/PageHeader";
import { MlbHomeRunBoard } from "@/components/markets/MlbHomeRunBoard";
import { getMlbHomeRunBoardSnapshot } from "@/lib/data/player-markets";
import { SPORTS } from "@/lib/markets-registry";

export const dynamic = "force-dynamic";

export default async function MarketsPage() {
  const snapshot = await getMlbHomeRunBoardSnapshot();
  const mlb = SPORTS.find((sport) => sport.slug === "mlb");
  const mlbMarkets = mlb?.markets.filter((m) => m.status === "live") ?? [];
  const otherSports = SPORTS.filter((s) => s.slug !== "mlb" && s.markets.some((m) => m.status === "live"));

  return (
    <div>
      <PageHeader
        title="MLB Home Runs"
        description="Daily probability-first batter HR candidates from projected lineups, probable pitchers, venue context, and recent hitter form."
        meta={`Candidate model · ${snapshot.slateDate} · ${snapshot.status}`}
      />

      <div className="mb-6 flex flex-wrap items-center gap-2 text-sm">
        <span className="font-medium text-muted-foreground">MLB:</span>
        <Link href="/markets/mlb/home-runs" className="rounded-md bg-secondary px-3 py-1.5 font-semibold text-foreground">
          Home runs
        </Link>
        {mlbMarkets.filter((m) => m.slug !== "home-runs").map((market) => (
          <Link
            key={market.slug}
            href={market.href}
            className="rounded-md px-3 py-1.5 font-medium text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground"
          >
            {market.short}
          </Link>
        ))}
        <span className="ml-4 font-medium text-muted-foreground">Other:</span>
        {otherSports.map((sport) => (
          <Link
            key={sport.slug}
            href={sport.markets[0]?.href ?? `/markets/${sport.slug}`}
            className="rounded-md px-3 py-1.5 font-medium text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground"
          >
            {sport.label}
          </Link>
        ))}
      </div>

      <MlbHomeRunBoard snapshot={snapshot} />
    </div>
  );
}
