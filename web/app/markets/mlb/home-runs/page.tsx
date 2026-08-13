import { PageHeader } from "@/components/dashboard/PageHeader";
import { MlbHomeRunBoard } from "@/components/markets/MlbHomeRunBoard";
import { getMlbHomeRunBoardSnapshot } from "@/lib/data/player-markets";

export const dynamic = "force-dynamic";

export default async function MlbHomeRunsPage() {
  const snapshot = await getMlbHomeRunBoardSnapshot();

  return (
    <div>
      <PageHeader
        title="MLB Home Runs"
        description="Daily probability-first batter HR candidates from projected lineups, probable pitchers, venue context, and recent hitter form."
        meta={`Candidate model · ${snapshot.slateDate} · ${snapshot.status}`}
      />

      <MlbHomeRunBoard snapshot={snapshot} />
    </div>
  );
}
