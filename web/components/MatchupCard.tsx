'use client';

import { useState } from 'react';
import type { TeamInfo } from '@/lib/bracketUtils';
import { ROUND_NAMES, REGION_NAMES } from '@/lib/bracketUtils';

interface MatchupCardProps {
  teamA: TeamInfo;
  teamB: TeamInfo;
  round: number;
  region: string;
  probTournament: number;
  probNeutral: number;
  winner?: TeamInfo | null;
  onAdvance: (team: TeamInfo) => void;
  onClose: () => void;
}

export function MatchupCard({
  teamA,
  teamB,
  round,
  region,
  probTournament,
  probNeutral,
  winner,
  onAdvance,
  onClose,
}: MatchupCardProps) {
  const [mode, setMode] = useState<'tournament' | 'neutral'>('tournament');

  const probA = mode === 'tournament' ? probTournament : probNeutral;
  const probB = 1 - probA;
  const pctA = (probA * 100).toFixed(1);
  const pctB = (probB * 100).toFixed(1);

  const seedGap = Math.abs(teamA.seed - teamB.seed);
  const tournDelta = Math.abs(probTournament - probNeutral);
  const seedInfluence = tournDelta > 0.01
    ? `Seed weighting shifts this matchup by ${(tournDelta * 100).toFixed(1)} percentage points`
    : 'Seed has minimal effect on this matchup';

  const roundLabel = ROUND_NAMES[round] ?? `Round ${round}`;
  const regionLabel = region === 'F4'
    ? 'Final Four'
    : `${REGION_NAMES[region] ?? region} Region`;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm anim-fade-in"
      onClick={onClose}
    >
      <div
        className="bg-card border border-border rounded-xl shadow-2xl w-full max-w-md mx-4 overflow-hidden anim-zoom-in"
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="px-5 pt-4 pb-3 border-b border-border/50 flex items-center justify-between bg-secondary/20">
          <div>
            <p className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">{regionLabel}</p>
            <p className="text-sm font-bold">{roundLabel}</p>
          </div>
          <button
            onClick={onClose}
            className="w-7 h-7 rounded-full flex items-center justify-center text-muted-foreground hover:text-foreground hover:bg-secondary transition-colors"
          >
            &times;
          </button>
        </div>

        {/* Matchup */}
        <div className="px-5 py-5">
          {/* Team names */}
          <div className="flex items-center justify-between mb-5">
            <div className="text-center flex-1">
              <span className="text-xs text-muted-foreground font-mono block mb-0.5">({teamA.seed})</span>
              <p className="text-lg font-bold leading-tight">{teamA.team}</p>
            </div>
            <div className="text-[10px] text-muted-foreground/50 font-bold px-4 py-1 rounded-full bg-secondary/30">VS</div>
            <div className="text-center flex-1">
              <span className="text-xs text-muted-foreground font-mono block mb-0.5">({teamB.seed})</span>
              <p className="text-lg font-bold leading-tight">{teamB.team}</p>
            </div>
          </div>

          {/* Probability bar */}
          <div className="mb-1">
            <div className="flex justify-between text-sm font-bold mb-1.5">
              <span className={probA >= 0.5 ? 'text-emerald-500' : 'text-muted-foreground'}>{pctA}%</span>
              <span className={probB >= 0.5 ? 'text-blue-500' : 'text-muted-foreground'}>{pctB}%</span>
            </div>
            <div className="h-4 rounded-full overflow-hidden flex bg-muted/50 shadow-inner">
              <div
                className="bg-gradient-to-r from-emerald-600 to-emerald-500 transition-all duration-500 ease-out rounded-l-full"
                style={{ width: `${probA * 100}%` }}
              />
              <div
                className="bg-gradient-to-r from-blue-500 to-blue-600 transition-all duration-500 ease-out rounded-r-full"
                style={{ width: `${probB * 100}%` }}
              />
            </div>
            <div className="flex justify-between text-[10px] text-muted-foreground mt-1">
              <span>{teamA.team}</span>
              <span>{teamB.team}</span>
            </div>
          </div>

          {/* Mode toggle */}
          <div className="flex items-center justify-center mt-5 mb-4">
            <div className="inline-flex rounded-lg border border-border overflow-hidden">
              <button
                onClick={() => setMode('tournament')}
                className={`px-4 py-2 text-xs font-medium transition-all duration-200 ${
                  mode === 'tournament'
                    ? 'bg-accent text-accent-foreground'
                    : 'bg-transparent text-muted-foreground hover:bg-secondary/50'
                }`}
              >
                Tournament
              </button>
              <button
                onClick={() => setMode('neutral')}
                className={`px-4 py-2 text-xs font-medium transition-all duration-200 border-l border-border ${
                  mode === 'neutral'
                    ? 'bg-accent text-accent-foreground'
                    : 'bg-transparent text-muted-foreground hover:bg-secondary/50'
                }`}
              >
                Neutral Court
              </button>
            </div>
          </div>

          {/* Stats row */}
          <div className="grid grid-cols-3 gap-2 text-center text-[11px]">
            <div className="p-2.5 rounded-lg bg-secondary/20 border border-border/50">
              <p className="text-muted-foreground mb-0.5">Seed Gap</p>
              <p className="font-bold text-base">{seedGap}</p>
            </div>
            <div className="p-2.5 rounded-lg bg-secondary/20 border border-border/50">
              <p className="text-muted-foreground mb-0.5">Tournament</p>
              <p className="font-bold text-base">{(probTournament * 100).toFixed(1)}%</p>
            </div>
            <div className="p-2.5 rounded-lg bg-secondary/20 border border-border/50">
              <p className="text-muted-foreground mb-0.5">Neutral</p>
              <p className="font-bold text-base">{(probNeutral * 100).toFixed(1)}%</p>
            </div>
          </div>
          <p className="text-[10px] text-muted-foreground text-center mt-2.5 italic">{seedInfluence}</p>
        </div>

        {/* Advance buttons */}
        <div className="px-5 pb-5 flex gap-3">
          <button
            onClick={() => onAdvance(teamA)}
            className={`flex-1 py-3 rounded-lg text-sm font-bold transition-all duration-200 border ${
              winner?.teamId === teamA.teamId
                ? 'bg-emerald-600 text-white border-emerald-500 shadow-lg shadow-emerald-500/20'
                : 'bg-secondary/30 text-foreground border-border hover:bg-emerald-600/20 hover:border-emerald-500/50'
            }`}
          >
            Advance {teamA.team}
          </button>
          <button
            onClick={() => onAdvance(teamB)}
            className={`flex-1 py-3 rounded-lg text-sm font-bold transition-all duration-200 border ${
              winner?.teamId === teamB.teamId
                ? 'bg-blue-600 text-white border-blue-500 shadow-lg shadow-blue-500/20'
                : 'bg-secondary/30 text-foreground border-border hover:bg-blue-600/20 hover:border-blue-500/50'
            }`}
          >
            Advance {teamB.team}
          </button>
        </div>
      </div>
    </div>
  );
}
