'use client';

import { useState, useMemo } from 'react';

export interface TeamProbabilities {
  R64: number;
  R32: number;
  S16: number;
  E8: number;
  F4: number;
  Championship: number;
  Winner: number;
}

export interface BracketTeam {
  team: string;
  teamId: number;
  seed: number;
  region: string;
  probabilities: TeamProbabilities;
}

interface BracketViewProps {
  teams: BracketTeam[];
  onTeamClick?: (team: BracketTeam) => void;
  selectedTeam?: string | null;
  constraints?: { team: string; action: string; round?: number }[];
}

const ROUND_LABELS = ['Round of 64', 'Round of 32', 'Sweet 16', 'Elite 8', 'Final Four', 'Championship', 'Champion'];
const ROUND_KEYS: (keyof TeamProbabilities)[] = ['R64', 'R32', 'S16', 'E8', 'F4', 'Championship', 'Winner'];
const REGION_NAMES: Record<string, string> = { E: 'East', W: 'West', S: 'South', M: 'Midwest' };
const SEED_ORDER = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15];

function getConfidenceColor(prob: number): string {
  if (prob >= 0.75) return 'bg-emerald-500/20 border-emerald-500/50 text-emerald-400';
  if (prob >= 0.55) return 'bg-blue-500/20 border-blue-500/50 text-blue-400';
  if (prob >= 0.45) return 'bg-amber-500/20 border-amber-500/50 text-amber-400';
  return 'bg-red-500/20 border-red-500/50 text-red-400';
}

function getWinProbBg(prob: number): string {
  if (prob >= 0.10) return 'bg-emerald-500';
  if (prob >= 0.05) return 'bg-blue-500';
  if (prob >= 0.02) return 'bg-amber-500';
  return 'bg-muted';
}

function MatchupCell({
  team,
  roundKey,
  isSelected,
  isConstrained,
  onClick,
}: {
  team: BracketTeam;
  roundKey: keyof TeamProbabilities;
  isSelected: boolean;
  isConstrained: boolean;
  onClick: () => void;
}) {
  const prob = team.probabilities[roundKey];
  const colorClass = getConfidenceColor(prob);

  return (
    <button
      onClick={onClick}
      className={`
        w-full px-2 py-1.5 rounded border text-left text-xs transition-all
        hover:ring-1 hover:ring-accent/50 cursor-pointer
        ${colorClass}
        ${isSelected ? 'ring-2 ring-accent' : ''}
        ${isConstrained ? 'opacity-40 line-through' : ''}
      `}
    >
      <div className="flex items-center justify-between gap-1">
        <span className="font-mono text-muted-foreground w-4 text-right text-[10px]">{team.seed}</span>
        <span className="font-medium truncate flex-1">{team.team}</span>
        <span className="font-mono text-[10px] tabular-nums">{(prob * 100).toFixed(0)}%</span>
      </div>
    </button>
  );
}

function RegionBracket({
  region,
  teams,
  selectedTeam,
  constraints,
  onTeamClick,
}: {
  region: string;
  teams: BracketTeam[];
  selectedTeam?: string | null;
  constraints: { team: string; action: string; round?: number }[];
  onTeamClick: (team: BracketTeam) => void;
}) {
  const sortedTeams = useMemo(() => {
    const seedMap = new Map(teams.map(t => [t.seed, t]));
    return SEED_ORDER.map(s => seedMap.get(s)).filter(Boolean) as BracketTeam[];
  }, [teams]);

  const constrainedTeams = new Set(
    constraints.filter(c => c.action === 'eliminate').map(c => c.team)
  );

  return (
    <div className="flex-1 min-w-[280px]">
      <h3 className="text-sm font-bold text-accent mb-2 uppercase tracking-wider">
        {REGION_NAMES[region] || region} Region
      </h3>
      <div className="space-y-0.5">
        {sortedTeams.map((team, i) => (
          <div key={team.teamId}>
            <MatchupCell
              team={team}
              roundKey="R32"
              isSelected={selectedTeam === team.team}
              isConstrained={constrainedTeams.has(team.team)}
              onClick={() => onTeamClick(team)}
            />
            {i % 2 === 1 && i < sortedTeams.length - 1 && (
              <div className="h-px bg-border/30 my-1" />
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

export function BracketView({ teams, onTeamClick, selectedTeam, constraints = [] }: BracketViewProps) {
  const [viewMode, setViewMode] = useState<'bracket' | 'table'>('bracket');
  const [sortKey, setSortKey] = useState<keyof TeamProbabilities>('Winner');
  const [sortAsc, setSortAsc] = useState(false);

  const regionTeams = useMemo(() => {
    const map: Record<string, BracketTeam[]> = {};
    for (const t of teams) {
      if (!map[t.region]) map[t.region] = [];
      map[t.region].push(t);
    }
    return map;
  }, [teams]);

  const sortedTeams = useMemo(() => {
    return [...teams].sort((a, b) => {
      const diff = b.probabilities[sortKey] - a.probabilities[sortKey];
      return sortAsc ? -diff : diff;
    });
  }, [teams, sortKey, sortAsc]);

  const handleSort = (key: keyof TeamProbabilities) => {
    if (key === sortKey) setSortAsc(!sortAsc);
    else { setSortKey(key); setSortAsc(false); }
  };

  return (
    <div className="space-y-4">
      {/* View toggle */}
      <div className="flex items-center gap-2">
        <button
          onClick={() => setViewMode('bracket')}
          className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
            viewMode === 'bracket' ? 'bg-accent text-accent-foreground' : 'bg-secondary text-secondary-foreground hover:bg-secondary/80'
          }`}
        >
          Bracket View
        </button>
        <button
          onClick={() => setViewMode('table')}
          className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
            viewMode === 'table' ? 'bg-accent text-accent-foreground' : 'bg-secondary text-secondary-foreground hover:bg-secondary/80'
          }`}
        >
          Table View
        </button>
      </div>

      {viewMode === 'bracket' ? (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
          {['E', 'W', 'S', 'M'].map(region => (
            <RegionBracket
              key={region}
              region={region}
              teams={regionTeams[region] || []}
              selectedTeam={selectedTeam}
              constraints={constraints}
              onTeamClick={t => onTeamClick?.(t)}
            />
          ))}
        </div>
      ) : (
        <div className="overflow-x-auto rounded-lg border border-border">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-border bg-secondary/50">
                <th className="text-left px-3 py-2 font-medium text-muted-foreground">Team</th>
                <th className="text-center px-2 py-2 font-medium text-muted-foreground w-10">Seed</th>
                <th className="text-center px-2 py-2 font-medium text-muted-foreground w-12">Rgn</th>
                {ROUND_KEYS.slice(1).map(key => (
                  <th
                    key={key}
                    className="text-center px-2 py-2 font-medium text-muted-foreground cursor-pointer hover:text-foreground w-16"
                    onClick={() => handleSort(key)}
                  >
                    {key}
                    {sortKey === key && (sortAsc ? ' ▲' : ' ▼')}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {sortedTeams.map((team, i) => {
                const isEliminated = constraints.some(c => c.team === team.team && c.action === 'eliminate');
                return (
                  <tr
                    key={team.teamId}
                    className={`
                      border-b border-border/50 cursor-pointer transition-colors
                      hover:bg-secondary/30
                      ${selectedTeam === team.team ? 'bg-accent/10' : ''}
                      ${isEliminated ? 'opacity-40' : ''}
                      ${i % 2 === 0 ? 'bg-card/30' : ''}
                    `}
                    onClick={() => onTeamClick?.(team)}
                  >
                    <td className={`px-3 py-1.5 font-medium ${isEliminated ? 'line-through' : ''}`}>
                      {team.team}
                    </td>
                    <td className="text-center px-2 py-1.5 font-mono text-muted-foreground">{team.seed}</td>
                    <td className="text-center px-2 py-1.5 text-muted-foreground">
                      {REGION_NAMES[team.region]?.[0] || team.region}
                    </td>
                    {ROUND_KEYS.slice(1).map(key => {
                      const prob = team.probabilities[key];
                      return (
                        <td key={key} className="text-center px-2 py-1.5">
                          <div className="flex items-center justify-center gap-1">
                            <div className={`h-1.5 rounded-full ${getWinProbBg(prob)}`}
                                 style={{ width: `${Math.max(prob * 100, 2)}%`, minWidth: '2px', maxWidth: '40px' }} />
                            <span className="font-mono tabular-nums text-[10px]">
                              {prob >= 0.01 ? `${(prob * 100).toFixed(1)}` : prob > 0 ? '<1' : '0'}
                            </span>
                          </div>
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
