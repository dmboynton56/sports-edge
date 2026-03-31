'use client';

import type { MatchupSlot, TeamInfo } from '@/lib/bracketUtils';
import { REGION_NAMES } from '@/lib/bracketUtils';

interface MiniFullBracketProps {
  regionBrackets: Record<string, MatchupSlot[][]>;
  finalFour: MatchupSlot[][];
  selectedRegion: string | null;
  onRegionClick: (region: string) => void;
}

const ABBRS: Record<string, string> = {
  'Duke': 'DUKE', 'UConn': 'UCON', 'Michigan State': 'MSU', 'Kansas': 'KU',
  "St. John's": 'SJU', 'Louisville': 'LOU', 'UCLA': 'UCLA', 'Ohio State': 'OSU',
  'TCU': 'TCU', 'UCF': 'UCF', 'South Florida': 'USF', 'Northern Iowa': 'UNI',
  'Cal Baptist': 'CBU', 'North Dakota State': 'NDSU', 'Furman': 'FUR', 'Siena': 'SNA',
  'Arizona': 'ARIZ', 'Purdue': 'PUR', 'Gonzaga': 'GONZ', 'Arkansas': 'ARK',
  'Wisconsin': 'WISC', 'BYU': 'BYU', 'Miami (FL)': 'MIA', 'Villanova': 'NOVA',
  'Utah State': 'USU', 'Missouri': 'MIZZ', 'NC State': 'NCST', 'High Point': 'HPU',
  'Hawaii': 'HAW', 'Kennesaw State': 'KSU', 'Queens (NC)': 'QU', 'Long Island': 'LIU',
  'Florida': 'FLA', 'Houston': 'HOU', 'Illinois': 'ILL', 'Nebraska': 'NEB',
  'Vanderbilt': 'VAN', 'North Carolina': 'UNC', "Saint Mary's": 'SMC', 'Clemson': 'CLEM',
  'Iowa': 'IOWA', 'Texas A&M': 'TAMU', 'VCU': 'VCU', 'McNeese': 'MCN',
  'Troy': 'TROY', 'Penn': 'PENN', 'Idaho': 'IDHO', 'Lehigh': 'LEH',
  'Michigan': 'MICH', 'Iowa State': 'ISU', 'Virginia': 'UVA', 'Alabama': 'BAMA',
  'Texas Tech': 'TTU', 'Tennessee': 'TENN', 'Kentucky': 'UK', 'Georgia': 'UGA',
  'Saint Louis': 'SLU', 'Santa Clara': 'SCU', 'SMU': 'SMU', 'Akron': 'AKR',
  'Hofstra': 'HOF', 'Wright State': 'WSU', 'Tennessee State': 'TSU', 'UMBC': 'UMBC',
};

function abbr(name: string): string {
  return ABBRS[name] ?? name.slice(0, 4).toUpperCase();
}

function RegionSummaryCard({
  region,
  rounds,
  isSelected,
  onClick,
}: {
  region: string;
  rounds: MatchupSlot[][];
  isSelected: boolean;
  onClick: () => void;
}) {
  const regionWinner = rounds[3]?.[0]?.winner;
  const picksInRegion = rounds.reduce(
    (acc, round) => acc + round.filter(s => s.winner).length,
    0,
  );
  const totalSlots = 8 + 4 + 2 + 1; // 15 matchups per region

  // Show the E8 matchup teams as a preview
  const e8 = rounds[3]?.[0];
  const s16teams = rounds[2]?.map(s => s.winner).filter(Boolean) as TeamInfo[];

  return (
    <button
      onClick={onClick}
      className={`p-3 rounded-lg border transition-all duration-200 text-left w-full ${
        isSelected
          ? 'border-accent bg-accent/5 shadow-sm'
          : 'border-border/60 hover:border-accent/40 bg-card/50'
      }`}
    >
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-bold">{REGION_NAMES[region]}</span>
        <span className="text-[10px] text-muted-foreground">
          {picksInRegion}/{totalSlots}
        </span>
      </div>

      {/* Show top seeds + region winner path */}
      <div className="space-y-1">
        {rounds[0]?.slice(0, 4).map(slot => {
          const winner = slot.winner;
          return (
            <div key={slot.slotId} className="flex items-center gap-1 text-[10px]">
              <span className="text-muted-foreground/50 w-8 text-right font-mono">
                {slot.teamA?.seed}v{slot.teamB?.seed}
              </span>
              {winner ? (
                <span className="text-emerald-400 font-medium truncate">
                  {abbr(winner.team)}
                </span>
              ) : (
                <span className="text-muted-foreground/30">---</span>
              )}
            </div>
          );
        })}
        {rounds[0] && rounds[0].length > 4 && (
          <div className="text-[9px] text-muted-foreground/40">
            +{rounds[0].length - 4} more
          </div>
        )}
      </div>

      {/* Region winner */}
      {regionWinner && (
        <div className="mt-2 pt-2 border-t border-border/30 flex items-center gap-1">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
          <span className="text-[11px] font-bold text-emerald-400">
            ({regionWinner.seed}) {regionWinner.team}
          </span>
        </div>
      )}
    </button>
  );
}

export function MiniFullBracket({
  regionBrackets,
  finalFour,
  selectedRegion,
  onRegionClick,
}: MiniFullBracketProps) {
  const regions = ['E', 'W', 'S', 'M'] as const;
  const champion = finalFour[1]?.[0]?.winner;

  return (
    <div className="p-4 rounded-lg border border-border bg-card">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-sm font-bold">Bracket Overview</h2>
        {champion && (
          <span className="text-sm font-bold text-emerald-400 bg-emerald-500/10 px-3 py-1 rounded-full flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
            Champion: ({champion.seed}) {champion.team}
          </span>
        )}
      </div>

      {/* 2x2 region grid + Final Four */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-3">
        {regions.map(r => {
          const rounds = regionBrackets[r];
          if (!rounds) return null;
          return (
            <RegionSummaryCard
              key={r}
              region={r}
              rounds={rounds}
              isSelected={selectedRegion === r}
              onClick={() => onRegionClick(r)}
            />
          );
        })}
      </div>

      {/* Final Four button */}
      <button
        onClick={() => onRegionClick('F4')}
        className={`w-full p-3 rounded-lg border transition-all duration-200 text-left ${
          selectedRegion === 'F4'
            ? 'border-accent bg-accent/5 shadow-sm'
            : 'border-border/60 hover:border-accent/40 bg-card/50'
        }`}
      >
        <div className="flex items-center justify-between">
          <span className="text-xs font-bold">Final Four &amp; Championship</span>
          <div className="flex gap-3">
            {finalFour[0]?.map(semi => (
              <div key={semi.slotId} className="flex items-center gap-1 text-[10px]">
                {semi.teamA ? (
                  <span className="text-foreground/70">({semi.teamA.seed}) {abbr(semi.teamA.team)}</span>
                ) : (
                  <span className="text-muted-foreground/30">TBD</span>
                )}
                <span className="text-muted-foreground/40">vs</span>
                {semi.teamB ? (
                  <span className="text-foreground/70">({semi.teamB.seed}) {abbr(semi.teamB.team)}</span>
                ) : (
                  <span className="text-muted-foreground/30">TBD</span>
                )}
                {semi.winner && (
                  <>
                    <span className="text-muted-foreground/40 mx-0.5">&rarr;</span>
                    <span className="text-emerald-400 font-medium">{abbr(semi.winner.team)}</span>
                  </>
                )}
              </div>
            ))}
          </div>
        </div>
      </button>
    </div>
  );
}
