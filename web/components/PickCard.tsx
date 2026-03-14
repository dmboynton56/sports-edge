import React from 'react';

export interface Game {
  id: string;
  league: string;
  homeTeam: string;
  awayTeam: string;
  gameTimeUtc: string;
}

export interface OddsSnapshot {
  book: string;
  market: string;
  line: number;
  price: number;
}

export interface ModelPrediction {
  predictedSpread: number;
  homeWinProb: number;
}

export interface EnrichedPick {
  id: string;
  game: Game;
  currentOdds: OddsSnapshot;
  prediction: ModelPrediction;
  edgePts: number;
}

export function PickCard({ pick }: { pick: EnrichedPick }) {
  const isPositiveEdge = pick.edgePts > 0;
  
  // formatting helper
  const formatLine = (line: number) => line > 0 ? `+${line}` : line;

  return (
    <div className="group relative overflow-hidden rounded-xl border border-border bg-card text-card-foreground shadow-sm transition-all hover:shadow-md">
      {/* Edge indicator top bar */}
      <div className={`h-2 w-full ${isPositiveEdge ? 'bg-accent' : 'bg-muted'}`} />
      
      <div className="p-5">
        <div className="flex justify-between items-center mb-4">
          <span className="text-xs font-semibold tracking-wider text-muted-foreground uppercase">
            {pick.game.league}
          </span>
          <span className="text-xs text-muted-foreground">
            {new Date(pick.game.gameTimeUtc).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}
          </span>
        </div>

        <div className="flex flex-col space-y-3 mb-6">
          <div className="flex justify-between items-center font-semibold text-lg">
            <span>{pick.game.awayTeam}</span>
            <span className="text-muted-foreground/50">@</span>
            <span>{pick.game.homeTeam}</span>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4 rounded-lg bg-secondary/50 p-4 mb-4">
          <div className="flex flex-col">
            <span className="text-xs text-muted-foreground mb-1 uppercase tracking-wide">Market Consensus</span>
            <span className="font-mono text-lg font-bold">
              {formatLine(pick.currentOdds.line)}
              <span className="text-sm font-normal text-muted-foreground ml-1">({formatLine(pick.currentOdds.price)})</span>
            </span>
          </div>
          <div className="flex flex-col items-end">
            <span className="text-xs text-muted-foreground mb-1 uppercase tracking-wide">Model Line</span>
            <span className="font-mono text-lg font-bold text-primary">
              {formatLine(pick.prediction.predictedSpread)}
            </span>
          </div>
        </div>

        <div className="flex items-center justify-between mt-4 pt-4 border-t border-border">
          <div className="flex items-center space-x-2">
            <span className="text-sm text-muted-foreground">Win Prob:</span>
            <span className="font-bold">{(pick.prediction.homeWinProb * 100).toFixed(1)}%</span>
          </div>
          
          <div className={`flex items-center px-3 py-1 rounded-full text-sm font-bold ${
            isPositiveEdge ? 'bg-accent/20 text-accent' : 'bg-muted text-muted-foreground'
          }`}>
            <span className="mr-1">E[V]</span>
            {formatLine(pick.edgePts)} pts
          </div>
        </div>
      </div>
    </div>
  );
}
