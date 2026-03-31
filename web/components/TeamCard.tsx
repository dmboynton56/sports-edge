'use client';

import type { BracketTeam, TeamProbabilities } from './BracketView';

const ROUND_LABELS: { key: keyof TeamProbabilities; label: string }[] = [
  { key: 'R32', label: 'R32' },
  { key: 'S16', label: 'Sweet 16' },
  { key: 'E8', label: 'Elite 8' },
  { key: 'F4', label: 'Final Four' },
  { key: 'Championship', label: 'Title Game' },
  { key: 'Winner', label: 'Champion' },
];

const REGION_NAMES: Record<string, string> = {
  E: 'East', W: 'West', S: 'South', M: 'Midwest',
};

function ProbBar({ prob, label, highlight }: { prob: number; label: string; highlight?: boolean }) {
  const pct = prob * 100;
  const barColor = highlight
    ? 'bg-accent'
    : pct >= 30 ? 'bg-emerald-500' : pct >= 10 ? 'bg-blue-500' : pct >= 3 ? 'bg-amber-500' : 'bg-muted-foreground/30';

  return (
    <div className="flex items-center gap-2">
      <span className="text-[10px] text-muted-foreground w-16 text-right">{label}</span>
      <div className="flex-1 h-4 bg-secondary/50 rounded overflow-hidden relative">
        <div
          className={`h-full ${barColor} rounded transition-all duration-500`}
          style={{ width: `${Math.max(pct, 0.5)}%` }}
        />
        <span className="absolute inset-0 flex items-center px-1.5 text-[10px] font-mono tabular-nums">
          {pct >= 1 ? `${pct.toFixed(1)}%` : pct > 0 ? '<1%' : '0%'}
        </span>
      </div>
    </div>
  );
}

interface TeamCardProps {
  team: BracketTeam;
  baselineProbabilities?: TeamProbabilities;
  onClose?: () => void;
}

export function TeamCard({ team, baselineProbabilities, onClose }: TeamCardProps) {
  const hasComparison = baselineProbabilities && (
    baselineProbabilities.Winner !== team.probabilities.Winner
  );

  return (
    <div className="rounded-lg border border-border bg-card p-4 space-y-3">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-lg font-bold">{team.team}</span>
            <span className="px-1.5 py-0.5 rounded bg-accent/10 text-accent text-xs font-bold">
              #{team.seed}
            </span>
          </div>
          <p className="text-xs text-muted-foreground">
            {REGION_NAMES[team.region] || team.region} Region
          </p>
        </div>
        {onClose && (
          <button
            onClick={onClose}
            className="text-muted-foreground hover:text-foreground transition-colors p-1"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        )}
      </div>

      {/* Win probability highlight */}
      <div className="flex items-center gap-4 p-3 rounded-md bg-secondary/50">
        <div className="text-center">
          <p className="text-2xl font-bold text-accent">
            {(team.probabilities.Winner * 100).toFixed(1)}%
          </p>
          <p className="text-[10px] text-muted-foreground uppercase">Win Title</p>
        </div>
        <div className="h-10 w-px bg-border" />
        <div className="text-center">
          <p className="text-2xl font-bold">
            {(team.probabilities.F4 * 100).toFixed(1)}%
          </p>
          <p className="text-[10px] text-muted-foreground uppercase">Final Four</p>
        </div>
        <div className="h-10 w-px bg-border" />
        <div className="text-center">
          <p className="text-2xl font-bold">
            {(team.probabilities.E8 * 100).toFixed(1)}%
          </p>
          <p className="text-[10px] text-muted-foreground uppercase">Elite 8</p>
        </div>
      </div>

      {/* Advancement probability bars */}
      <div className="space-y-1.5">
        <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
          Advancement Probabilities
        </h4>
        {ROUND_LABELS.map(({ key, label }) => (
          <ProbBar
            key={key}
            prob={team.probabilities[key]}
            label={label}
            highlight={key === 'Winner'}
          />
        ))}
      </div>

      {/* Comparison with baseline */}
      {hasComparison && baselineProbabilities && (
        <div className="space-y-1.5 pt-2 border-t border-border">
          <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
            vs. Baseline
          </h4>
          <div className="grid grid-cols-3 gap-2 text-center">
            {(['F4', 'Championship', 'Winner'] as const).map(key => {
              const current = team.probabilities[key];
              const baseline = baselineProbabilities[key];
              const diff = current - baseline;
              const diffStr = diff > 0 ? `+${(diff * 100).toFixed(1)}` : (diff * 100).toFixed(1);
              const diffColor = diff > 0 ? 'text-emerald-400' : diff < 0 ? 'text-red-400' : 'text-muted-foreground';

              return (
                <div key={key} className="p-2 rounded bg-secondary/30">
                  <p className="text-[10px] text-muted-foreground">
                    {key === 'Championship' ? 'Title' : key}
                  </p>
                  <p className={`text-sm font-bold ${diffColor}`}>{diffStr}%</p>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
