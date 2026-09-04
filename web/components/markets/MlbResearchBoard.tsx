"use client";

import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { EmptyState } from "@/components/dashboard/EmptyState";
import type { MlbResearchBoardData, MlbResearchPrediction } from "@/lib/data/mlb-research";

type MlbResearchBoardProps = {
  data: MlbResearchBoardData;
  title: string;
  description: string;
};

function formatProbability(prob: number | null | undefined): string {
  if (prob == null) return "—";
  return `${(prob * 100).toFixed(1)}%`;
}

function formatPrice(price: number | null | undefined): string {
  if (price == null) return "—";
  return price > 0 ? `+${price}` : `${price}`;
}

function formatSide(side: string | null | undefined): string {
  if (!side) return "";
  return side.charAt(0).toUpperCase() + side.slice(1);
}

function MoneylineRow({ prediction }: { prediction: MlbResearchPrediction }) {
  const hasPrices = prediction.oddsStatus === "ok";
  
  return (
    <div className="grid grid-cols-12 gap-3 py-3 border-b last:border-0">
      <div className="col-span-4 space-y-1">
        <div className="font-medium">{prediction.awayTeam} @ {prediction.homeTeam}</div>
        {prediction.venue && <div className="text-xs text-muted-foreground">{prediction.venue}</div>}
      </div>
      <div className="col-span-3 space-y-1 text-sm">
        <div>Home: {formatProbability(prediction.homeWinProb)}</div>
        <div>Away: {formatProbability(prediction.awayWinProb)}</div>
      </div>
      <div className="col-span-3 space-y-1 text-sm">
        {hasPrices ? (
          <>
            <div>Home: {formatPrice(prediction.homePrice)}</div>
            <div>Away: {formatPrice(prediction.awayPrice)}</div>
          </>
        ) : (
          <div className="text-muted-foreground italic">Model only</div>
        )}
      </div>
      <div className="col-span-2 text-sm">
        {hasPrices && prediction.edge != null ? (
          <Badge variant={prediction.edge > 0.03 ? "default" : "outline"}>
            {prediction.recommendedSide ? `${formatSide(prediction.recommendedSide)} ` : ""}Edge: {(prediction.edge * 100).toFixed(1)}%
          </Badge>
        ) : (
          <span className="text-muted-foreground">—</span>
        )}
      </div>
    </div>
  );
}

function RunLineRow({ prediction }: { prediction: MlbResearchPrediction }) {
  const hasPrices = prediction.oddsStatus === "ok";
  const homeLine = prediction.homeRunlineLine ?? -1.5;
  const awayLine = -homeLine;
  
  return (
    <div className="grid grid-cols-12 gap-3 py-3 border-b last:border-0">
      <div className="col-span-4 space-y-1">
        <div className="font-medium">{prediction.awayTeam} @ {prediction.homeTeam}</div>
        {prediction.venue && <div className="text-xs text-muted-foreground">{prediction.venue}</div>}
      </div>
      <div className="col-span-3 space-y-1 text-sm">
        <div>Projected margin: {prediction.predictedMargin?.toFixed(1) ?? "—"}</div>
        <div>
          {hasPrices && prediction.recommendedSide
            ? `${formatSide(prediction.recommendedSide)} cover`
            : "Home -1.5 cover"}: {formatProbability(
              hasPrices ? prediction.recommendedProbability : prediction.pHomeCover15,
            )}
        </div>
      </div>
      <div className="col-span-3 space-y-1 text-sm">
        {hasPrices ? (
          <>
            <div>Home {homeLine > 0 ? "+" : ""}{homeLine}: {formatPrice(prediction.homeRunlinePrice)}</div>
            <div>Away {awayLine > 0 ? "+" : ""}{awayLine}: {formatPrice(prediction.awayRunlinePrice)}</div>
          </>
        ) : (
          <div className="text-muted-foreground italic">Model only</div>
        )}
      </div>
      <div className="col-span-2 text-sm">
        {hasPrices && prediction.edge != null ? (
          <Badge variant={prediction.edge > 0.03 ? "default" : "outline"}>
            {prediction.recommendedSide ? `${formatSide(prediction.recommendedSide)} ` : ""}Edge: {(prediction.edge * 100).toFixed(1)}%
          </Badge>
        ) : (
          <span className="text-muted-foreground">—</span>
        )}
      </div>
    </div>
  );
}

function TotalsRow({ prediction }: { prediction: MlbResearchPrediction }) {
  const hasPrices = prediction.oddsStatus === "ok";
  const displayLine = prediction.totalLine ?? 8.5;
  
  return (
    <div className="grid grid-cols-12 gap-3 py-3 border-b last:border-0">
      <div className="col-span-4 space-y-1">
        <div className="font-medium">{prediction.awayTeam} @ {prediction.homeTeam}</div>
        {prediction.venue && <div className="text-xs text-muted-foreground">{prediction.venue}</div>}
      </div>
      <div className="col-span-3 space-y-1 text-sm">
        <div>Projected: {prediction.predictedTotal?.toFixed(1) ?? "—"}</div>
        <div>{prediction.recommendedSide ? `${formatSide(prediction.recommendedSide)} ${displayLine}` : `Over ${displayLine}`}: {formatProbability(prediction.recommendedProbability)}</div>
      </div>
      <div className="col-span-3 space-y-1 text-sm">
        {hasPrices ? (
          <>
            <div>Over {displayLine}: {formatPrice(prediction.overPrice)}</div>
            <div>Under {displayLine}: {formatPrice(prediction.underPrice)}</div>
          </>
        ) : (
          <div className="text-muted-foreground italic">Model only</div>
        )}
      </div>
      <div className="col-span-2 text-sm">
        {hasPrices && prediction.edge != null ? (
          <Badge variant={prediction.edge > 0.03 ? "default" : "outline"}>
            {prediction.recommendedSide ? `${formatSide(prediction.recommendedSide)} ` : ""}Edge: {(prediction.edge * 100).toFixed(1)}%
          </Badge>
        ) : (
          <span className="text-muted-foreground">—</span>
        )}
      </div>
    </div>
  );
}

export function MlbResearchBoard({ data, title, description }: MlbResearchBoardProps) {
  const RowComponent = 
    data.market === "moneyline" ? MoneylineRow :
    data.market === "run_line" ? RunLineRow :
    TotalsRow;

  return (
    <Card className="p-5">
      <div className="mb-4 space-y-2">
        <div className="flex items-center gap-2">
          <h2 className="text-lg font-semibold">{title}</h2>
          <Badge variant="outline">Research</Badge>
        </div>
        <p className="text-sm text-muted-foreground">{description}</p>
        {data.generatedAt && (
          <p className="text-xs text-muted-foreground">
            Generated: {new Date(data.generatedAt).toLocaleString()}
          </p>
        )}
      </div>

      {data.gaps.length > 0 && (
        <div className="mb-4 rounded-md bg-yellow-50 p-3 text-sm text-yellow-900 dark:bg-yellow-950 dark:text-yellow-200">
          <ul className="list-disc list-inside space-y-1">
            {data.gaps.map((gap, index) => (
              <li key={index}>{gap}</li>
            ))}
          </ul>
        </div>
      )}

      {data.predictions.length === 0 ? (
        <EmptyState
          className="border-0 bg-transparent"
          title="No predictions today"
          description={`Research ${data.market} predictions will appear here when today's slate is scored.`}
        />
      ) : (
        <div className="space-y-0">
          <div className="grid grid-cols-12 gap-3 pb-2 border-b font-medium text-sm text-muted-foreground">
            <div className="col-span-4">Matchup</div>
            <div className="col-span-3">Model Probability</div>
            <div className="col-span-3">Sportsbook Price</div>
            <div className="col-span-2">Edge</div>
          </div>
          {data.predictions.map((prediction) => (
            <RowComponent key={prediction.id} prediction={prediction} />
          ))}
        </div>
      )}
    </Card>
  );
}
