'use client';

import React, { useState } from 'react';

export interface PlayerPrediction {
  name: string;
  score: number;
  expectedSG: number;
  models: {
    ridge: number;
    rf: number;
    lgbm: number;
    xgb: number;
    nn: number;
  };
  mcWin: number;
  mcTop5: number;
  mcTop10: number;
  clsTop10: number;
  clsTop20: number;
  clsWin: number;
}

function formatScore(score: number) {
  if (score === 0) return 'E';
  return score > 0 ? `+${score}` : `${score}`;
}

function sgColor(sg: number) {
  if (sg > 0.5) return 'text-emerald-400';
  if (sg > 0) return 'text-emerald-400/70';
  if (sg > -0.5) return 'text-amber-400';
  return 'text-red-400';
}

function probCell(val: number, threshold?: number) {
  const pct = val * 100;
  const bold = threshold && pct > threshold;
  return (
    <span className={bold ? 'text-emerald-400 font-semibold' : 'text-muted-foreground'}>
      {pct.toFixed(1)}%
    </span>
  );
}

function SGSparkBar({ value, max = 2.5 }: { value: number; max?: number }) {
  const absVal = Math.min(Math.abs(value), max);
  const pct = (absVal / max) * 100;
  const positive = value >= 0;
  return (
    <div className="flex items-center gap-1.5 w-full">
      <div className="flex-1 h-2 bg-secondary/50 rounded-full overflow-hidden relative">
        <div className="absolute left-1/2 top-0 w-px h-full bg-border/50" />
        <div
          className={`absolute top-0 h-full rounded-full ${positive ? 'bg-emerald-500/80' : 'bg-red-500/80'}`}
          style={{
            width: `${pct / 2}%`,
            left: positive ? '50%' : `${50 - pct / 2}%`,
          }}
        />
      </div>
      <span className={`text-xs font-mono w-12 text-right ${sgColor(value)}`}>
        {value >= 0 ? '+' : ''}{value.toFixed(2)}
      </span>
    </div>
  );
}

export function PlayersLeaderboard({ players }: { players: PlayerPrediction[] }) {
  const [expandedIdx, setExpandedIdx] = useState<number | null>(null);
  const [sortKey, setSortKey] = useState<'score' | 'expectedSG' | 'mcWin'>('score');

  const sorted = [...players].sort((a, b) => {
    if (sortKey === 'score') return a.score - b.score;
    if (sortKey === 'expectedSG') return b.expectedSG - a.expectedSG;
    return b.mcWin - a.mcWin;
  });

  return (
    <div className="rounded-xl border border-border bg-card overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border bg-secondary/30">
              <th className="text-left px-4 py-3 font-medium text-muted-foreground w-8">#</th>
              <th className="text-left px-4 py-3 font-medium text-muted-foreground">Player</th>
              <th
                className="text-right px-4 py-3 font-medium text-muted-foreground cursor-pointer hover:text-foreground transition-colors"
                onClick={() => setSortKey('score')}
              >
                R3{sortKey === 'score' && ' ▼'}
              </th>
              <th
                className="text-right px-4 py-3 font-medium text-muted-foreground cursor-pointer hover:text-foreground transition-colors"
                onClick={() => setSortKey('expectedSG')}
              >
                Exp SG/R{sortKey === 'expectedSG' && ' ▼'}
              </th>
              <th className="px-4 py-3 font-medium text-muted-foreground text-center w-32">SG Visual</th>
              <th
                className="text-right px-4 py-3 font-medium text-muted-foreground cursor-pointer hover:text-foreground transition-colors"
                onClick={() => setSortKey('mcWin')}
              >
                Win%{sortKey === 'mcWin' && ' ▼'}
              </th>
              <th className="text-right px-4 py-3 font-medium text-muted-foreground">Top 5</th>
              <th className="text-right px-4 py-3 font-medium text-muted-foreground">Top 10</th>
              <th className="text-right px-4 py-3 font-medium text-muted-foreground hidden lg:table-cell">Cls T10</th>
              <th className="text-right px-4 py-3 font-medium text-muted-foreground hidden lg:table-cell">Cls T20</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((p, i) => {
              const isExpanded = expandedIdx === i;
              const isLeader = i === 0 && sortKey === 'score';
              return (
                <React.Fragment key={p.name}>
                  <tr
                    className={`border-b border-border/50 cursor-pointer transition-colors hover:bg-secondary/20 ${isLeader ? 'bg-emerald-500/5' : ''}`}
                    onClick={() => setExpandedIdx(isExpanded ? null : i)}
                  >
                    <td className="px-4 py-3 text-muted-foreground font-mono text-xs">{i + 1}</td>
                    <td className="px-4 py-3 font-medium">
                      <div className="flex items-center gap-2">
                        {p.name}
                        {p.expectedSG > 0.7 && (
                          <span className="text-[10px] bg-emerald-500/20 text-emerald-400 px-1.5 py-0.5 rounded font-bold">HOT</span>
                        )}
                        {p.expectedSG < -2 && (
                          <span className="text-[10px] bg-red-500/20 text-red-400 px-1.5 py-0.5 rounded font-bold">FADE</span>
                        )}
                      </div>
                    </td>
                    <td className="px-4 py-3 text-right font-mono font-bold">{formatScore(p.score)}</td>
                    <td className={`px-4 py-3 text-right font-mono font-bold ${sgColor(p.expectedSG)}`}>
                      {p.expectedSG >= 0 ? '+' : ''}{p.expectedSG.toFixed(3)}
                    </td>
                    <td className="px-4 py-3"><SGSparkBar value={p.expectedSG} /></td>
                    <td className="px-4 py-3 text-right font-mono font-bold">
                      {probCell(p.mcWin, 10)}
                    </td>
                    <td className="px-4 py-3 text-right font-mono">{probCell(p.mcTop5, 30)}</td>
                    <td className="px-4 py-3 text-right font-mono">{probCell(p.mcTop10, 50)}</td>
                    <td className="px-4 py-3 text-right font-mono hidden lg:table-cell">{probCell(p.clsTop10, 15)}</td>
                    <td className="px-4 py-3 text-right font-mono hidden lg:table-cell">{probCell(p.clsTop20, 25)}</td>
                  </tr>
                  {isExpanded && (
                    <tr className="bg-secondary/10">
                      <td colSpan={10} className="px-6 py-4">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                          <div>
                            <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3">
                              Individual Model SG Predictions
                            </h4>
                            <div className="space-y-2">
                              {([
                                ['Ridge', p.models.ridge, 'bg-slate-400'],
                                ['Random Forest', p.models.rf, 'bg-amber-500'],
                                ['LightGBM', p.models.lgbm, 'bg-blue-500'],
                                ['XGBoost', p.models.xgb, 'bg-cyan-500'],
                                ['Neural Net', p.models.nn, 'bg-purple-500'],
                              ] as [string, number, string][]).map(([label, val, color]) => (
                                <div key={label} className="flex items-center gap-3">
                                  <span className="text-xs text-muted-foreground w-24 shrink-0 text-right">{label}</span>
                                  <div className="flex-1 h-4 bg-secondary/50 rounded-full overflow-hidden relative">
                                    <div className="absolute left-1/2 top-0 w-px h-full bg-border/50" />
                                    <div
                                      className={`absolute top-0 h-full rounded-full ${color}`}
                                      style={{
                                        width: `${Math.min(Math.abs(val) / 2.5, 1) * 50}%`,
                                        left: val >= 0 ? '50%' : `${50 - Math.min(Math.abs(val) / 2.5, 1) * 50}%`,
                                        opacity: 0.8,
                                      }}
                                    />
                                  </div>
                                  <span className={`text-xs font-mono font-bold w-14 text-right ${sgColor(val)}`}>
                                    {val >= 0 ? '+' : ''}{val.toFixed(3)}
                                  </span>
                                </div>
                              ))}
                              <div className="border-t border-dashed border-border pt-2 mt-1">
                                <div className="flex items-center gap-3">
                                  <span className="text-xs font-bold text-muted-foreground w-24 shrink-0 text-right">Meta Stack</span>
                                  <div className="flex-1 h-4 bg-secondary/50 rounded-full overflow-hidden relative">
                                    <div className="absolute left-1/2 top-0 w-px h-full bg-border/50" />
                                    <div
                                      className="absolute top-0 h-full rounded-full bg-emerald-500"
                                      style={{
                                        width: `${Math.min(Math.abs(p.expectedSG) / 2.5, 1) * 50}%`,
                                        left: p.expectedSG >= 0 ? '50%' : `${50 - Math.min(Math.abs(p.expectedSG) / 2.5, 1) * 50}%`,
                                        opacity: 0.9,
                                      }}
                                    />
                                  </div>
                                  <span className={`text-xs font-mono font-bold w-14 text-right ${sgColor(p.expectedSG)}`}>
                                    {p.expectedSG >= 0 ? '+' : ''}{p.expectedSG.toFixed(3)}
                                  </span>
                                </div>
                              </div>
                            </div>
                          </div>
                          <div>
                            <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3">
                              Monte Carlo Probabilities (50K sims)
                            </h4>
                            <div className="grid grid-cols-3 gap-3">
                              {([
                                ['Win', p.mcWin],
                                ['Top 5', p.mcTop5],
                                ['Top 10', p.mcTop10],
                              ] as [string, number][]).map(([label, val]) => (
                                <div key={label} className="rounded-lg bg-secondary/50 p-3 text-center">
                                  <div className="text-[10px] text-muted-foreground uppercase mb-1">{label}</div>
                                  <div className={`text-lg font-bold font-mono ${val > 0.1 ? 'text-emerald-400' : 'text-foreground'}`}>
                                    {(val * 100).toFixed(1)}%
                                  </div>
                                </div>
                              ))}
                            </div>
                            <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mt-4 mb-3">
                              Classifier Probabilities (LR, best calibrated)
                            </h4>
                            <div className="grid grid-cols-3 gap-3">
                              {([
                                ['Top 10', p.clsTop10],
                                ['Top 20', p.clsTop20],
                                ['Win', p.clsWin],
                              ] as [string, number][]).map(([label, val]) => (
                                <div key={label} className="rounded-lg bg-secondary/50 p-3 text-center">
                                  <div className="text-[10px] text-muted-foreground uppercase mb-1">{label}</div>
                                  <div className="text-lg font-bold font-mono text-foreground">
                                    {(val * 100).toFixed(1)}%
                                  </div>
                                </div>
                              ))}
                            </div>
                          </div>
                        </div>
                      </td>
                    </tr>
                  )}
                </React.Fragment>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
