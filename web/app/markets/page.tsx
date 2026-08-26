import Link from "next/link";
import { MlbHomeRunBoard } from "@/components/markets/MlbHomeRunBoard";
import { getMlbHomeRunBoardSnapshot } from "@/lib/data/player-markets";
import { SPORTS } from "@/lib/markets-registry";
import { formatDateTime, formatNumber, formatPct } from "@/lib/format";

export const dynamic = "force-dynamic";

export default async function MarketsPage() {
  const snapshot = await getMlbHomeRunBoardSnapshot();
  const mlb = SPORTS.find((sport) => sport.slug === "mlb");
  const mlbMarkets = mlb?.markets.filter((m) => m.status === "live") ?? [];
  const otherSports = SPORTS.filter((s) => s.slug !== "mlb" && s.markets.some((m) => m.status === "live"));

  return (
    <div>
      <div className="mb-6 border-b border-border pb-4">
        <h1 className="text-lg font-medium">MLB Home Runs</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Daily probability-first batter HR candidates · {snapshot.slateDate} · {snapshot.status}
        </p>
      </div>

      <div className="mb-6 flex flex-wrap items-center gap-x-4 gap-y-2 text-xs">
        <span className="tag">mlb</span>
        <Link href="/markets/mlb/home-runs" className="font-medium text-foreground">
          Home runs
        </Link>
        {mlbMarkets.filter((m) => m.slug !== "home-runs").map((market) => (
          <Link
            key={market.slug}
            href={market.href}
            className="text-muted-foreground transition-colors hover:text-foreground"
          >
            {market.short}
          </Link>
        ))}
        <span className="tag ml-2">other</span>
        {otherSports.map((sport) => (
          <Link
            key={sport.slug}
            href={sport.markets[0]?.href ?? `/markets/${sport.slug}`}
            className="text-muted-foreground transition-colors hover:text-foreground"
          >
            {sport.label}
          </Link>
        ))}
      </div>

      <div className="mb-6 flex gap-8 border-b border-border pb-4">
        <div>
          <div className="tag">candidates</div>
          <div className="num mt-1 text-2xl">{formatNumber(snapshot.counts.candidates)}</div>
        </div>
        <div>
          <div className="tag">priced</div>
          <div className="num mt-1 text-2xl">{formatNumber(snapshot.counts.priced)}</div>
        </div>
        <div>
          <div className="tag">top-25 coverage</div>
          <div className="num mt-1 text-2xl">{formatPct(snapshot.counts.top25Coverage)}</div>
        </div>
        <div>
          <div className="tag">refreshed</div>
          <div className="num mt-1 text-sm">{formatDateTime(snapshot.completedAt)}</div>
        </div>
      </div>

      <MlbHomeRunBoard snapshot={snapshot} />
    </div>
  );
}
