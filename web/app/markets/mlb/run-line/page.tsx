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
        description="Run-line cover probabilities from the MLB v1 margin-residual research model. Model-only rows show when sportsbook prices are unavailable."
        meta={data.generatedAt ? `Generated ${data.generatedAt}` : data.slateDate}
      />

      <MlbResearchBoard
        data={data}
        title="Run Line Board"
        description="Research probabilities are translated to the currently posted home/away line using held-out margin residuals. This is not a production-validated Trusted market."
      />
    </div>
  );
}
