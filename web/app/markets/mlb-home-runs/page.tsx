import { PageHeader } from "@/components/dashboard/PageHeader";
import { MlbHomeRunBoard } from "@/components/markets/MlbHomeRunBoard";
import { getMlbHomeRunBoardData, getMlbHomeRunModelLabel } from "@/lib/data/player-markets";

export default async function MlbHomeRunsPage() {
  const board = await getMlbHomeRunBoardData();
  const modelSummary = board.availableModels
    .map((modelKey) => getMlbHomeRunModelLabel(modelKey))
    .join(" · ");

  return (
    <div>
      <PageHeader
        title="MLB Home Runs"
        description="Daily probability-first batter HR candidates from projected lineups, probable pitchers, venue context, and recent hitter form."
        meta={[board.generatedAt, modelSummary].filter(Boolean).join(" · ")}
      />

      <MlbHomeRunBoard board={board} />
    </div>
  );
}
