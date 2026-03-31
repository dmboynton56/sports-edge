'use client';

import type { MatchupSlot, TeamInfo } from '@/lib/bracketUtils';
import { ROUND_NAMES, REGION_NAMES } from '@/lib/bracketUtils';

interface RegionDetailProps {
  region: string;
  rounds: MatchupSlot[][];
  onPickWinner: (slot: MatchupSlot, winner: TeamInfo) => void;
  onMatchupClick: (slot: MatchupSlot) => void;
  getProb: (a: number, b: number) => number;
}

function TeamSlot({
  team,
  isWinner,
  isClickable,
  onClick,
  side,
}: {
  team: TeamInfo | null;
  isWinner: boolean;
  isClickable: boolean;
  onClick: () => void;
  side: 'top' | 'bottom';
}) {
  if (!team) {
    return (
      <div className={`h-9 flex items-center px-2.5 text-[11px] text-muted-foreground/40 italic
        ${side === 'top' ? 'border-b border-border/30' : ''}`}>
        TBD
      </div>
    );
  }

  return (
    <button
      onClick={onClick}
      disabled={!isClickable}
      className={`h-9 w-full flex items-center gap-1.5 px-2.5 text-left transition-all duration-200
        ${side === 'top' ? 'border-b border-border/30' : ''}
        ${isClickable ? 'hover:bg-accent/10 cursor-pointer' : 'cursor-default'}
        ${isWinner ? 'bg-emerald-500/15 font-bold' : ''}
      `}
    >
      <span className="text-[10px] font-mono text-muted-foreground w-5 text-right shrink-0">
        {team.seed}
      </span>
      <span className={`text-[12px] truncate ${isWinner ? 'text-emerald-400' : ''}`}>
        {team.team}
      </span>
    </button>
  );
}

function MatchupBox({
  slot,
  onPickWinner,
  onMatchupClick,
  getProb,
  roundIdx,
}: {
  slot: MatchupSlot;
  onPickWinner: (slot: MatchupSlot, winner: TeamInfo) => void;
  onMatchupClick: (slot: MatchupSlot) => void;
  getProb: (a: number, b: number) => number;
  roundIdx: number;
}) {
  const { teamA, teamB, winner } = slot;
  const hasBoth = teamA !== null && teamB !== null;
  const prob = hasBoth ? getProb(teamA.teamId, teamB.teamId) : null;

  return (
    <div className="relative group">
      <div className={`
        border border-border/60 rounded-md overflow-hidden bg-card/80 w-[180px]
        transition-all duration-200
        ${hasBoth ? 'hover:border-accent/40 hover:shadow-sm' : ''}
        ${winner ? 'border-emerald-500/30' : ''}
      `}>
        <TeamSlot
          team={teamA}
          isWinner={!!winner && winner.teamId === teamA?.teamId}
          isClickable={hasBoth}
          onClick={() => teamA && onPickWinner(slot, teamA)}
          side="top"
        />
        <TeamSlot
          team={teamB}
          isWinner={!!winner && winner.teamId === teamB?.teamId}
          isClickable={hasBoth}
          onClick={() => teamB && onPickWinner(slot, teamB)}
          side="bottom"
        />
      </div>
      {hasBoth && prob !== null && (
        <button
          onClick={() => onMatchupClick(slot)}
          className="absolute -right-1 top-1/2 -translate-y-1/2 translate-x-full
            bg-card border border-border rounded-full px-2 py-0.5 text-[9px] font-bold
            text-accent hover:bg-accent/10 hover:border-accent/50 transition-all duration-200
            cursor-pointer z-10 whitespace-nowrap shadow-sm"
          title="Click for game details"
        >
          {(prob * 100).toFixed(0)}%
        </button>
      )}
      {/* Right connector line to next round */}
      {roundIdx < 3 && (
        <div className="absolute top-1/2 -right-[17px] w-4 border-t border-border/30" />
      )}
    </div>
  );
}

export function RegionDetail({
  region,
  rounds,
  onPickWinner,
  onMatchupClick,
  getProb,
}: RegionDetailProps) {
  const isFinalFour = region === 'F4';
  const regionLabel = isFinalFour
    ? 'Final Four & Championship'
    : `${REGION_NAMES[region] ?? region} Region`;

  const lastRound = rounds[rounds.length - 1];
  const regionWinner = lastRound?.[0]?.winner;

  // Height of a single matchup box in px
  const MATCHUP_H = 72; // 2 x 36px team slots
  const BASE_GAP = 8; // px gap in R64

  return (
    <div className="p-5 rounded-lg border border-border bg-card">
      <div className="flex items-center justify-between mb-5">
        <h3 className="text-sm font-bold tracking-tight">{regionLabel}</h3>
        {regionWinner && (
          <span className="text-xs font-bold text-emerald-400 flex items-center gap-1.5 bg-emerald-500/10 px-2.5 py-1 rounded-full">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
            ({regionWinner.seed}) {regionWinner.team}
          </span>
        )}
      </div>

      <div className="overflow-x-auto pb-2">
        <div className="flex gap-12 min-w-max items-start">
          {rounds.map((matchups, roundIdx) => {
            const roundName = isFinalFour
              ? ROUND_NAMES[roundIdx + 4] ?? `Round ${roundIdx + 4}`
              : ROUND_NAMES[roundIdx] ?? `Round ${roundIdx}`;

            // Calculate vertical spacing to align matchups with their "source" matchups
            const spacingMultiplier = Math.pow(2, roundIdx);
            const gapPx = BASE_GAP + (spacingMultiplier - 1) * (MATCHUP_H + BASE_GAP);
            const topPadPx = roundIdx === 0 ? 0 :
              (Math.pow(2, roundIdx) - 1) * (MATCHUP_H + BASE_GAP) / 2;

            return (
              <div key={roundIdx} className="flex flex-col items-center shrink-0">
                <p className="text-[10px] text-muted-foreground uppercase tracking-wider mb-3 font-semibold">
                  {roundName}
                </p>
                <div
                  className="flex flex-col"
                  style={{
                    gap: `${gapPx}px`,
                    paddingTop: `${topPadPx}px`,
                  }}
                >
                  {matchups.map(slot => (
                    <MatchupBox
                      key={slot.slotId}
                      slot={slot}
                      onPickWinner={onPickWinner}
                      onMatchupClick={onMatchupClick}
                      getProb={getProb}
                      roundIdx={roundIdx}
                    />
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
