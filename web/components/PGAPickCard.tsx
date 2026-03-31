import React from 'react';

export interface PGATournament {
  id: string;
  name: string;
  location: string;
  round: number;
}

export interface ModelBreakdown {
  xgboostSG: number;       // XGBoost predicted Expected SG per round
  pytorchSG: number;       // PyTorch Course Fit predicted SG per round
  ensembleSG: number;      // Final blended SG used in simulation
  pytorchAdjustment: number; // The delta PyTorch applied (positive = boost, negative = fade)
  xgboostConfidence: number; // 0-1 how confident XGBoost is (based on feature availability)
  pytorchCourseFit: number;  // -1 to +1 how well this player fits this course per PyTorch
  keyDrivers: string[];      // Top 3 reasons for the prediction
}

export interface PGAPrediction {
  playerName: string;
  currentScore: number;
  winProb: number;
  top5Prob: number;
  top10Prob: number;
  top20Prob: number;
  impliedOdds: number;
}

export interface PGAEnrichedPick {
  id: string;
  tournament: PGATournament;
  prediction: PGAPrediction;
  modelBreakdown: ModelBreakdown;
  edgePct: number;
}

function SGBar({ label, value, color, maxRange = 3 }: { label: string; value: number; color: string; maxRange?: number }) {
  const pct = Math.min(Math.abs(value) / maxRange, 1) * 100;
  const isPositive = value >= 0;
  
  return (
    <div className="flex items-center gap-3">
      <span className="text-xs text-muted-foreground w-20 shrink-0 text-right">{label}</span>
      <div className="flex-1 h-5 bg-secondary/50 rounded-full overflow-hidden relative">
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="w-px h-full bg-border" />
        </div>
        <div 
          className={`absolute top-0 h-full rounded-full transition-all duration-500 ${color}`}
          style={{
            width: `${pct / 2}%`,
            left: isPositive ? '50%' : `${50 - pct / 2}%`,
          }}
        />
      </div>
      <span className={`text-xs font-mono font-bold w-12 ${value >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
        {value >= 0 ? '+' : ''}{value.toFixed(2)}
      </span>
    </div>
  );
}

function CourseFitMeter({ value }: { value: number }) {
  const pct = ((value + 1) / 2) * 100; // map -1..+1 to 0..100
  const label = value > 0.3 ? 'Strong Fit' : value > 0 ? 'Mild Fit' : value > -0.3 ? 'Mild Mismatch' : 'Poor Fit';
  const color = value > 0.3 ? 'text-emerald-400' : value > 0 ? 'text-emerald-400/60' : value > -0.3 ? 'text-amber-400' : 'text-red-400';
  
  return (
    <div className="flex flex-col gap-1">
      <div className="flex justify-between items-center">
        <span className="text-[10px] text-muted-foreground uppercase tracking-wide">Course Fit (PyTorch)</span>
        <span className={`text-xs font-bold ${color}`}>{label}</span>
      </div>
      <div className="h-2 bg-secondary/50 rounded-full overflow-hidden">
        <div 
          className="h-full rounded-full transition-all duration-700"
          style={{
            width: `${pct}%`,
            background: `linear-gradient(90deg, hsl(0 70% 50%), hsl(45 80% 55%) 50%, hsl(142 71% 45%))`,
          }}
        />
      </div>
    </div>
  );
}

export function PGAPickCard({ pick, defaultExpanded = false }: { pick: PGAEnrichedPick; defaultExpanded?: boolean }) {
  const [expanded, setExpanded] = React.useState(defaultExpanded);
  const isPositiveEdge = pick.edgePct > 0;
  const bd = pick.modelBreakdown;
  
  const formatProb = (prob: number) => `${(prob * 100).toFixed(1)}%`;
  const formatScore = (score: number) => score === 0 ? 'E' : score > 0 ? `+${score}` : score.toString();

  return (
    <div className="group relative overflow-hidden rounded-xl border border-border bg-card text-card-foreground shadow-sm transition-all hover:shadow-md">
      <div className={`h-1.5 w-full ${isPositiveEdge ? 'bg-accent' : 'bg-muted'}`} />
      
      <div className="p-5">
        {/* Header */}
        <div className="flex justify-between items-center mb-3">
          <span className="text-xs font-semibold tracking-wider text-muted-foreground uppercase truncate max-w-[150px]">
            {pick.tournament.name}
          </span>
          <span className="text-xs text-muted-foreground bg-secondary px-2 py-0.5 rounded-md">
            R{pick.tournament.round}
          </span>
        </div>

        {/* Player + Score */}
        <div className="flex justify-between items-center mb-5">
          <div className="flex flex-col">
            <span className="font-bold text-xl">{pick.prediction.playerName}</span>
            <span className="text-sm text-muted-foreground">
              Score: <span className="font-medium text-foreground">{formatScore(pick.prediction.currentScore)}</span>
            </span>
          </div>
          <div className={`flex items-center px-3 py-1 rounded-full text-sm font-bold ${
            isPositiveEdge ? 'bg-accent/20 text-accent' : 'bg-red-500/20 text-red-400'
          }`}>
            {pick.edgePct > 0 ? '+' : ''}{(pick.edgePct * 100).toFixed(1)}% EV
          </div>
        </div>

        {/* Win Probability vs Market */}
        <div className="grid grid-cols-2 gap-4 rounded-lg bg-secondary/50 p-4 mb-4">
          <div className="flex flex-col">
            <span className="text-xs text-muted-foreground mb-1 uppercase tracking-wide">Market (Win)</span>
            <span className="font-mono text-lg font-bold text-muted-foreground">
              {formatProb(pick.prediction.impliedOdds)}
            </span>
          </div>
          <div className="flex flex-col items-end">
            <span className="text-xs text-muted-foreground mb-1 uppercase tracking-wide">Ensemble (Win)</span>
            <span className={`font-mono text-lg font-bold ${isPositiveEdge ? 'text-accent' : ''}`}>
              {formatProb(pick.prediction.winProb)}
            </span>
          </div>
        </div>

        {/* Placement Probabilities */}
        <div className="grid grid-cols-3 gap-2 mb-4 text-center border-t border-b border-border py-3">
          <div className="flex flex-col">
            <span className="text-[10px] text-muted-foreground uppercase">Top 5</span>
            <span className="font-medium text-sm">{formatProb(pick.prediction.top5Prob)}</span>
          </div>
          <div className="flex flex-col border-x border-border">
            <span className="text-[10px] text-muted-foreground uppercase">Top 10</span>
            <span className="font-medium text-sm">{formatProb(pick.prediction.top10Prob)}</span>
          </div>
          <div className="flex flex-col">
            <span className="text-[10px] text-muted-foreground uppercase">Top 20</span>
            <span className="font-medium text-sm">{formatProb(pick.prediction.top20Prob)}</span>
          </div>
        </div>

        {/* Expand/Collapse Toggle */}
        <button 
          onClick={() => setExpanded(!expanded)} 
          className="w-full flex items-center justify-center gap-2 text-xs text-muted-foreground hover:text-foreground transition-colors py-1"
        >
          <span>{expanded ? 'Hide' : 'Show'} Model Breakdown</span>
          <svg 
            className={`w-3 h-3 transition-transform duration-200 ${expanded ? 'rotate-180' : ''}`} 
            fill="none" viewBox="0 0 24 24" stroke="currentColor"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </button>

        {/* Model Breakdown Panel */}
        {expanded && (
          <div className="mt-4 pt-4 border-t border-border space-y-5 animate-in fade-in slide-in-from-top-2 duration-300">
            
            {/* SG Predictions by Model */}
            <div>
              <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3">
                Expected Strokes Gained (per round)
              </h4>
              <div className="space-y-2.5">
                <SGBar label="XGBoost" value={bd.xgboostSG} color="bg-blue-500" />
                <SGBar label="PyTorch" value={bd.pytorchSG} color="bg-purple-500" />
                <div className="border-t border-dashed border-border pt-2">
                  <SGBar label="Ensemble" value={bd.ensembleSG} color="bg-emerald-500" />
                </div>
              </div>
            </div>

            {/* PyTorch Adjustment Callout */}
            <div className={`rounded-lg p-3 text-sm ${
              bd.pytorchAdjustment >= 0 
                ? 'bg-purple-500/10 border border-purple-500/20' 
                : 'bg-amber-500/10 border border-amber-500/20'
            }`}>
              <div className="flex items-center gap-2 mb-1">
                <span className="text-[10px] font-bold uppercase tracking-wider text-purple-400">PyTorch Course Fit Adjustment</span>
              </div>
              <p className="text-muted-foreground text-xs leading-relaxed">
                {bd.pytorchAdjustment >= 0 ? (
                  <>The deep learning model <span className="text-purple-400 font-bold">boosted</span> this player by <span className="font-mono font-bold text-purple-400">+{bd.pytorchAdjustment.toFixed(2)} SG</span> because their learned player embedding shows a strong historical fit with this course&apos;s characteristics.</>
                ) : (
                  <>The deep learning model <span className="text-amber-400 font-bold">faded</span> this player by <span className="font-mono font-bold text-amber-400">{bd.pytorchAdjustment.toFixed(2)} SG</span> because their learned embedding suggests a poor interaction with this course&apos;s layout and conditions.</>
                )}
              </p>
            </div>

            {/* Course Fit Meter */}
            <CourseFitMeter value={bd.pytorchCourseFit} />

            {/* Key Drivers */}
            <div>
              <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">
                Key Prediction Drivers
              </h4>
              <div className="space-y-1.5">
                {bd.keyDrivers.map((driver, i) => (
                  <div key={i} className="flex items-start gap-2 text-xs text-muted-foreground">
                    <span className="text-accent mt-0.5">&#9679;</span>
                    <span>{driver}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
