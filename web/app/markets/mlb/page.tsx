import { ChannelCard } from "@/components/dashboard/ChannelCard";
import { EmptyState } from "@/components/dashboard/EmptyState";
import { MarketsTable } from "@/components/dashboard/MarketsTable";
import { PageHeader, SectionHeading } from "@/components/dashboard/PageHeader";
import { MlbEvaluationSummary } from "@/components/markets/MlbEvaluationSummary";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { getMlbVerticalSummary } from "@/lib/data/mlb-vertical";
import { getProductionPredictionFeed } from "@/lib/data/player-markets";
import { getSport } from "@/lib/markets-registry";

export const dynamic = "force-dynamic";

const WINNER_MARKETS = new Set(["winner", "moneyline", "money_line", "ml"]);

export default async function MlbMarketsPage() {
  const [feed, evaluation] = await Promise.all([
    getProductionPredictionFeed(),
    getMlbVerticalSummary(),
  ]);
  const mlb = getSport("mlb");
  const predictions = feed.predictions.filter(
    (prediction) =>
      prediction.sport.toLowerCase() === "mlb" &&
      WINNER_MARKETS.has(prediction.market.toLowerCase()),
  );

  return (
    <div>
      <PageHeader
        title="MLB Markets"
        description="Choose a baseball market or inspect live team winner probabilities."
        meta={feed.generatedAt}
      />

      <div className="grid gap-3 sm:grid-cols-2">
        {mlb?.markets.map((market) => (
          <ChannelCard
            key={market.slug}
            sport="mlb"
            href={market.href}
            title={market.label}
            description={market.description}
            cta={`Open ${market.short}`}
            badge={
              market.status === "scaffold" ? (
                <Badge variant="outline">Scaffold</Badge>
              ) : null
            }
          />
        ))}
      </div>

      <SectionHeading
        title="Model evaluation and edges"
        note={evaluation?.as_of_date ? `as of ${evaluation.as_of_date}` : "Awaiting the next evaluation artifact"}
      />
      {evaluation ? (
        <MlbEvaluationSummary summary={evaluation} />
      ) : (
        <Card className="p-5">
          <EmptyState
            className="border-0 bg-transparent"
            title="Evaluation artifact not published yet"
            description="Run the MLB vertical evaluator to publish held-out metrics, free-odds coverage, and statistical edge rows."
          />
        </Card>
      )}

      <SectionHeading title="Winner board" note="Pre-live team moneyline probabilities" />
      <Card className="overflow-hidden p-5">
        {predictions.length > 0 ? (
          <MarketsTable initialPredictions={predictions} initialGaps={feed.gaps} />
        ) : (
          <EmptyState
            className="border-0 bg-transparent"
            title="No winner board today"
            description="Team moneyline probabilities publish once the day's schedule and prices are in. The home-run board above runs on its own cadence."
          />
        )}
      </Card>
    </div>
  );
}
