import { PageHeader } from "@/components/dashboard/PageHeader";
import { MlbHomeRunBoard } from "@/components/markets/MlbHomeRunBoard";
import { getMlbHomeRunBoardData, getMlbHomeRunModelLabel } from "@/lib/data/player-markets";

export const dynamic = "force-dynamic";

export default async function MlbHomeRunsPage() {
  const board = await getMlbHomeRunBoardData();
  const modelSummary = board.availableModels
    .map((modelKey) => getMlbHomeRunModelLabel(modelKey))
    .join(" · ");
  const sourceSummary = board.dataSource.replace(/_/g, " ");

  return (
    <div>
      <PageHeader
        title="MLB Home Runs"
        description="Daily probability-first batter HR candidates from projected lineups, probable pitchers, venue context, and recent hitter form."
        meta={[board.generatedAt, modelSummary, sourceSummary].filter(Boolean).join(" · ")}
      />

      <MlbHomeRunBoard board={board} />
    </div>
  );
}
