'use client';

import { useState } from 'react';
import type { BracketTeam } from './BracketView';

export interface SimConstraint {
  team: string;
  action: 'eliminate' | 'advance_to';
  round: number;
}

const ROUND_OPTIONS = [
  { value: 0, label: 'Round of 64' },
  { value: 1, label: 'Round of 32' },
  { value: 2, label: 'Sweet 16' },
  { value: 3, label: 'Elite 8' },
  { value: 4, label: 'Final Four' },
  { value: 5, label: 'Championship' },
];

interface SimulationControlsProps {
  teams: BracketTeam[];
  constraints: SimConstraint[];
  onConstraintsChange: (constraints: SimConstraint[]) => void;
  onRunSimulation: () => void;
  isRunning?: boolean;
  simCount: number;
  onSimCountChange: (count: number) => void;
  selectedTeam?: BracketTeam | null;
}

export function SimulationControls({
  teams,
  constraints,
  onConstraintsChange,
  onRunSimulation,
  isRunning = false,
  simCount,
  onSimCountChange,
  selectedTeam,
}: SimulationControlsProps) {
  const [showAddForm, setShowAddForm] = useState(false);
  const [newTeam, setNewTeam] = useState('');
  const [newAction, setNewAction] = useState<'eliminate' | 'advance_to'>('eliminate');
  const [newRound, setNewRound] = useState(0);

  const addConstraint = () => {
    const teamName = newTeam || selectedTeam?.team;
    if (!teamName) return;
    const exists = constraints.some(c => c.team === teamName && c.action === newAction && c.round === newRound);
    if (exists) return;

    onConstraintsChange([
      ...constraints,
      { team: teamName, action: newAction, round: newRound },
    ]);
    setNewTeam('');
    setShowAddForm(false);
  };

  const removeConstraint = (index: number) => {
    onConstraintsChange(constraints.filter((_, i) => i !== index));
  };

  const clearAll = () => onConstraintsChange([]);

  const addQuickConstraint = (team: string, action: 'eliminate' | 'advance_to', round: number) => {
    const exists = constraints.some(c => c.team === team && c.action === action && c.round === round);
    if (!exists) {
      onConstraintsChange([...constraints, { team, action, round }]);
    }
  };

  return (
    <div className="space-y-4">
      {/* Simulation settings */}
      <div className="flex items-center gap-4 flex-wrap">
        <div className="flex items-center gap-2">
          <label className="text-xs text-muted-foreground">Simulations:</label>
          <select
            value={simCount}
            onChange={e => onSimCountChange(Number(e.target.value))}
            className="bg-secondary border border-border rounded px-2 py-1 text-xs"
          >
            <option value={1000}>1,000</option>
            <option value={10000}>10,000</option>
            <option value={50000}>50,000</option>
            <option value={100000}>100,000</option>
          </select>
        </div>

        <button
          onClick={onRunSimulation}
          disabled={isRunning}
          className={`
            px-4 py-1.5 rounded-md text-xs font-bold transition-all
            ${isRunning
              ? 'bg-muted text-muted-foreground cursor-wait'
              : 'bg-accent text-accent-foreground hover:bg-accent/90 active:scale-95'
            }
          `}
        >
          {isRunning ? (
            <span className="flex items-center gap-2">
              <span className="h-3 w-3 border-2 border-current border-t-transparent rounded-full animate-spin" />
              Running...
            </span>
          ) : (
            `Run ${simCount.toLocaleString()} Simulations`
          )}
        </button>
      </div>

      {/* Quick actions for selected team */}
      {selectedTeam && (
        <div className="p-3 rounded-lg border border-accent/30 bg-accent/5">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-xs text-muted-foreground">Quick actions for</span>
            <span className="text-sm font-bold text-accent">
              ({selectedTeam.seed}) {selectedTeam.team}
            </span>
          </div>
          <div className="flex flex-wrap gap-2">
            {ROUND_OPTIONS.map(r => (
              <button
                key={`elim-${r.value}`}
                onClick={() => addQuickConstraint(selectedTeam.team, 'eliminate', r.value)}
                className="px-2 py-1 rounded text-[10px] bg-red-500/10 border border-red-500/30 text-red-400 hover:bg-red-500/20 transition-colors"
              >
                Lose {r.label}
              </button>
            ))}
            {ROUND_OPTIONS.slice(1).map(r => (
              <button
                key={`adv-${r.value}`}
                onClick={() => addQuickConstraint(selectedTeam.team, 'advance_to', r.value)}
                className="px-2 py-1 rounded text-[10px] bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 hover:bg-emerald-500/20 transition-colors"
              >
                Reach {r.label}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Active constraints */}
      {constraints.length > 0 && (
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
              Active Constraints ({constraints.length})
            </h4>
            <button
              onClick={clearAll}
              className="text-[10px] text-destructive hover:underline"
            >
              Clear All
            </button>
          </div>
          <div className="flex flex-wrap gap-1.5">
            {constraints.map((c, i) => (
              <div
                key={i}
                className={`
                  flex items-center gap-1.5 px-2 py-1 rounded-full text-[10px] font-medium
                  ${c.action === 'eliminate'
                    ? 'bg-red-500/10 border border-red-500/30 text-red-400'
                    : 'bg-emerald-500/10 border border-emerald-500/30 text-emerald-400'
                  }
                `}
              >
                <span>
                  {c.team} — {c.action === 'eliminate' ? 'Lose' : 'Reach'}{' '}
                  {ROUND_OPTIONS[c.round]?.label || `R${c.round}`}
                </span>
                <button
                  onClick={() => removeConstraint(i)}
                  className="hover:text-foreground transition-colors ml-0.5"
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Preset scenarios */}
      <div className="space-y-2">
        <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
          Preset Scenarios
        </h4>
        <div className="flex flex-wrap gap-2">
          <button
            onClick={() => {
              const oneSeeds = teams.filter(t => t.seed === 1);
              const newConstraints = oneSeeds.map(t => ({
                team: t.team, action: 'eliminate' as const, round: 1,
              }));
              onConstraintsChange(newConstraints);
            }}
            className="px-3 py-1.5 rounded-md text-[10px] bg-secondary border border-border hover:border-accent/50 transition-colors"
          >
            All #1 Seeds Lose R32
          </button>
          <button
            onClick={() => {
              const chalkConstraints = teams
                .filter(t => t.seed <= 4)
                .map(t => ({ team: t.team, action: 'advance_to' as const, round: 3 }));
              onConstraintsChange(chalkConstraints);
            }}
            className="px-3 py-1.5 rounded-md text-[10px] bg-secondary border border-border hover:border-accent/50 transition-colors"
          >
            All Top 4 Seeds to E8 (Chalk)
          </button>
          <button
            onClick={() => onConstraintsChange([])}
            className="px-3 py-1.5 rounded-md text-[10px] bg-secondary border border-border hover:border-accent/50 transition-colors"
          >
            Reset to Baseline
          </button>
        </div>
      </div>

      {/* Custom constraint form */}
      {showAddForm ? (
        <div className="p-3 rounded-lg border border-border bg-card space-y-2">
          <div className="grid grid-cols-3 gap-2">
            <div>
              <label className="text-[10px] text-muted-foreground">Team</label>
              <select
                value={newTeam}
                onChange={e => setNewTeam(e.target.value)}
                className="w-full bg-secondary border border-border rounded px-2 py-1 text-xs"
              >
                <option value="">Select team...</option>
                {teams
                  .sort((a, b) => a.seed - b.seed || a.team.localeCompare(b.team))
                  .map(t => (
                    <option key={t.teamId} value={t.team}>
                      ({t.seed}) {t.team}
                    </option>
                  ))}
              </select>
            </div>
            <div>
              <label className="text-[10px] text-muted-foreground">Action</label>
              <select
                value={newAction}
                onChange={e => setNewAction(e.target.value as 'eliminate' | 'advance_to')}
                className="w-full bg-secondary border border-border rounded px-2 py-1 text-xs"
              >
                <option value="eliminate">Eliminate in</option>
                <option value="advance_to">Advance to</option>
              </select>
            </div>
            <div>
              <label className="text-[10px] text-muted-foreground">Round</label>
              <select
                value={newRound}
                onChange={e => setNewRound(Number(e.target.value))}
                className="w-full bg-secondary border border-border rounded px-2 py-1 text-xs"
              >
                {ROUND_OPTIONS.map(r => (
                  <option key={r.value} value={r.value}>{r.label}</option>
                ))}
              </select>
            </div>
          </div>
          <div className="flex gap-2">
            <button
              onClick={addConstraint}
              className="px-3 py-1 rounded text-xs bg-accent text-accent-foreground font-medium"
            >
              Add
            </button>
            <button
              onClick={() => setShowAddForm(false)}
              className="px-3 py-1 rounded text-xs bg-secondary text-secondary-foreground"
            >
              Cancel
            </button>
          </div>
        </div>
      ) : (
        <button
          onClick={() => setShowAddForm(true)}
          className="text-xs text-accent hover:underline"
        >
          + Add custom constraint
        </button>
      )}
    </div>
  );
}
