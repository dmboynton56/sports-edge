import { PageHeader } from "@/components/dashboard/PageHeader";
import { MlbResearchBoard } from "@/components/markets/MlbResearchBoard";
import { getMlbResearchBoard } from "@/lib/data/mlb-research";

export const dynamic = "force-dynamic";

export default async function MlbRunLinePage() {
  const data = await getMlbResearchBoard("run_line");

  return (
    <div>
      <PageHeader
        title="MLB Run Line (Research)"
        description="Home -1.5 cover probabilities from the MLB run-line v1 research model. Model-only rows show when sportsbook prices are unavailable."
        meta={data.generatedAt ? `Generated ${data.generatedAt}` : data.slateDate}
      />

      <MlbResearchBoard
        data={data}
        title="Run Line Board (Home -1.5)"
        description="Research model probabilities for home -1.5 / away +1.5 run-line markets. This is a research board, not a production-validated Trusted market."
      />
    </div>
  );
}
