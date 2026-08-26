import { PageHeader } from "@/components/dashboard/PageHeader";
import { MlbResearchBoard } from "@/components/markets/MlbResearchBoard";
import { getMlbResearchBoard } from "@/lib/data/mlb-research";

export const dynamic = "force-dynamic";

export default async function MlbTotalsPage() {
  const data = await getMlbResearchBoard("total");

  return (
    <div>
      <PageHeader
        title="MLB Totals (Research)"
        description="Projected total runs and over/under probabilities from the MLB totals v1 research model. Model-only rows show when sportsbook prices are unavailable."
        meta={data.generatedAt ? `Generated ${data.generatedAt}` : data.slateDate}
      />

      <MlbResearchBoard
        data={data}
        title="Totals Board (O/U 8.5 & 9.5)"
        description="Research model projections for total runs and over/under probabilities. This is a research board, not a production-validated Trusted market."
      />
    </div>
  );
}
