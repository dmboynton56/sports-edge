'use client';

import React, { useState } from 'react';

export interface KeyFeature {
  feature: string;
  contrib: number;
  type: 'positive' | 'negative';
}

export interface RecentTournamentRow {
  tournament: string;
  start: string;
  position: string;
  scoreToPar: string;
  r1: number | null;
  r2: number | null;
  r3: number | null;
  r4: number | null;
  total: number | null;
}

export interface MarketOddsDetail {
  marketImplied: number;
  bestPrice: number;
  bestBook: string;
  edge: number;
  ev: number;
  kelly: number;
  bookOdds?: Record<string, { price: number; implied: number; decimal: number }>;
}

export interface PlayerOdds extends MarketOddsDetail {
  otherMarkets?: Record<string, MarketOddsDetail>;
}

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
  keyFeatures?: KeyFeature[];
  odds?: PlayerOdds;
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

function formatAmericanOdds(price: number) {
  return price > 0 ? `+${price}` : `${price}`;
}

function edgeBadge(edgeVal: number) {
  const pct = (edgeVal * 100).toFixed(1);
  if (edgeVal > 0.02) {
    return <span className="text-emerald-400 font-bold">+{pct}%</span>;
  }
  if (edgeVal < -0.02) {
    return <span className="text-red-400 font-bold">{pct}%</span>;
  }
  return <span className="text-muted-foreground">{pct}%</span>;
}

function evBadge(evVal: number) {
  if (evVal > 0) {
    return <span className="text-emerald-400 font-semibold">+{evVal.toFixed(2)}</span>;
  }
  return <span className="text-muted-foreground">{evVal.toFixed(2)}</span>;
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

export function PlayersLeaderboard({
  players,
  recentByPlayer = {},
}: {
  players: PlayerPrediction[];
  recentByPlayer?: Record<string, RecentTournamentRow[]>;
}) {
  const [expandedPlayer, setExpandedPlayer] = useState<string | null>(null);
  const [sortKey, setSortKey] = useState<'score' | 'expectedSG' | 'mcWin' | 'edge'>('score');

  const hasOdds = players.some(p => p.odds != null);

  const sorted = [...players].sort((a, b) => {
    if (sortKey === 'score') return a.score - b.score;
    if (sortKey === 'expectedSG') return b.expectedSG - a.expectedSG;
    if (sortKey === 'edge') return (b.odds?.edge ?? -999) - (a.odds?.edge ?? -999);
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
                To Par{sortKey === 'score' && ' ▼'}
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
              {hasOdds && (
                <>
                  <th className="text-right px-4 py-3 font-medium text-muted-foreground">Market</th>
                  <th
                    className="text-right px-4 py-3 font-medium text-muted-foreground cursor-pointer hover:text-foreground transition-colors"
                    onClick={() => setSortKey('edge')}
                  >
                    Edge{sortKey === 'edge' && ' ▼'}
                  </th>
                  <th className="text-right px-4 py-3 font-medium text-muted-foreground hidden lg:table-cell">E[V]</th>
                  <th className="text-right px-4 py-3 font-medium text-muted-foreground hidden lg:table-cell">Best Price</th>
                </>
              )}
              <th className="text-right px-4 py-3 font-medium text-muted-foreground">Top 5</th>
              <th className="text-right px-4 py-3 font-medium text-muted-foreground">Top 10</th>
              <th className="text-right px-4 py-3 font-medium text-muted-foreground hidden lg:table-cell">Cls T10</th>
              <th className="text-right px-4 py-3 font-medium text-muted-foreground hidden lg:table-cell">Cls T20</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((p, i) => {
              const isExpanded = expandedPlayer === p.name;
              const isLeader = i === 0 && sortKey === 'score';
              const recent = recentByPlayer[p.name] ?? [];
              return (
                <React.Fragment key={p.name}>
                  <tr
                    className={`border-b border-border/50 cursor-pointer transition-colors hover:bg-secondary/20 ${isLeader ? 'bg-emerald-500/5' : ''}`}
                    onClick={() => setExpandedPlayer(isExpanded ? null : p.name)}
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
                    {hasOdds && (
                      <>
                        <td className="px-4 py-3 text-right font-mono text-muted-foreground">
                          {p.odds ? `${(p.odds.marketImplied * 100).toFixed(1)}%` : '—'}
                        </td>
                        <td className="px-4 py-3 text-right font-mono">
                          {p.odds ? edgeBadge(p.odds.edge) : '—'}
                        </td>
                        <td className="px-4 py-3 text-right font-mono hidden lg:table-cell">
                          {p.odds ? evBadge(p.odds.ev) : '—'}
                        </td>
                        <td className="px-4 py-3 text-right font-mono text-xs hidden lg:table-cell">
                          {p.odds ? (
                            <span title={p.odds.bestBook}>
                              {formatAmericanOdds(p.odds.bestPrice)}
                              <span className="text-muted-foreground ml-1">({p.odds.bestBook.slice(0, 3).toUpperCase()})</span>
                            </span>
                          ) : '—'}
                        </td>
                      </>
                    )}
                    <td className="px-4 py-3 text-right font-mono">{probCell(p.mcTop5, 30)}</td>
                    <td className="px-4 py-3 text-right font-mono">{probCell(p.mcTop10, 50)}</td>
                    <td className="px-4 py-3 text-right font-mono hidden lg:table-cell">{probCell(p.clsTop10, 15)}</td>
                    <td className="px-4 py-3 text-right font-mono hidden lg:table-cell">{probCell(p.clsTop20, 25)}</td>
                  </tr>
                  {isExpanded && (
                    <tr className="bg-secondary/10">
                      <td colSpan={hasOdds ? 14 : 10} className="px-6 py-4">
                        <div className={`grid grid-cols-1 gap-6 ${p.odds ? 'md:grid-cols-4' : 'md:grid-cols-3'}`}>
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
                            <p className="text-[10px] text-muted-foreground mb-2 leading-snug">
                              LR win% can disagree with MC when features sit outside typical PGA training mix; use MC for ranks and sizing.
                            </p>
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
                          {p.odds && (
                            <div>
                              <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3">
                                Market Odds vs Model
                              </h4>
                              {(() => {
                                const allMarkets: { label: string; key: string; detail: MarketOddsDetail; modelPct: string }[] = [
                                  { label: 'Win', key: 'win', detail: p.odds!, modelPct: `${(p.mcWin * 100).toFixed(1)}%` },
                                ];
                                if (p.odds!.otherMarkets) {
                                  const labels: Record<string, string> = { top5: 'Top 5', top10: 'Top 10', top20: 'Top 20', made_cut: 'Make Cut' };
                                  for (const [mkt, detail] of Object.entries(p.odds!.otherMarkets)) {
                                    allMarkets.push({ label: labels[mkt] ?? mkt, key: mkt, detail, modelPct: `${(detail.edge + detail.marketImplied) * 100 | 0}%~` });
                                  }
                                }
                                return (
                                  <div className="space-y-3">
                                    {allMarkets.map(({ label, key, detail }) => (
                                      <div key={key} className={`rounded-lg p-3 text-sm ${
                                        detail.edge > 0.02 ? 'bg-emerald-500/10 border border-emerald-500/20' : 'bg-secondary/30 border border-border'
                                      }`}>
                                        <div className="flex justify-between items-center mb-1.5">
                                          <span className="text-[10px] font-bold uppercase tracking-wider">{label}</span>
                                          <span className="text-xs font-mono font-bold">
                                            {formatAmericanOdds(detail.bestPrice)}{' '}
                                            <span className="text-muted-foreground font-normal">({detail.bestBook})</span>
                                          </span>
                                        </div>
                                        <div className="grid grid-cols-3 gap-2">
                                          <div className="text-center">
                                            <div className="text-[10px] text-muted-foreground">Market</div>
                                            <div className="font-mono text-xs">{(detail.marketImplied * 100).toFixed(1)}%</div>
                                          </div>
                                          <div className="text-center">
                                            <div className="text-[10px] text-muted-foreground">Edge</div>
                                            <div className="font-mono text-xs">{edgeBadge(detail.edge)}</div>
                                          </div>
                                          <div className="text-center">
                                            <div className="text-[10px] text-muted-foreground">E[V]</div>
                                            <div className="font-mono text-xs">{evBadge(detail.ev)}</div>
                                          </div>
                                        </div>
                                      </div>
                                    ))}
                                  </div>
                                );
                              })()}
                              {p.odds.bookOdds && Object.keys(p.odds.bookOdds).length > 1 && (
                                <div className="mt-3">
                                  <div className="text-[10px] text-muted-foreground uppercase mb-1.5">Win — Line Shopping</div>
                                  <div className="space-y-1">
                                    {Object.entries(p.odds.bookOdds)
                                      .sort(([, a], [, b]) => b.decimal - a.decimal)
                                      .map(([book, info]) => (
                                        <div key={book} className="flex justify-between items-center text-xs">
                                          <span className="text-muted-foreground capitalize">{book}</span>
                                          <span className={`font-mono ${book === p.odds!.bestBook ? 'text-emerald-400 font-bold' : ''}`}>
                                            {formatAmericanOdds(info.price)}
                                          </span>
                                        </div>
                                      ))}
                                  </div>
                                </div>
                              )}
                            </div>
                          )}
                          {p.keyFeatures && p.keyFeatures.length > 0 && (
                            <div>
                              <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3">
                                Model Reasoning (Top Features)
                              </h4>
                              <div className="space-y-2">
                                {p.keyFeatures.map((kf, idx) => (
                                  <div key={idx} className="flex items-center gap-3">
                                    <div className="flex-1 min-w-0">
                                      <p className="text-xs font-medium text-foreground truncate" title={kf.feature}>
                                        {kf.feature}
                                      </p>
                                    </div>
                                    <span
                                      className={`text-[10px] font-bold px-1.5 py-0.5 rounded uppercase tracking-wider ${
                                        kf.type === 'positive'
                                          ? 'bg-emerald-500/20 text-emerald-400'
                                          : 'bg-red-500/20 text-red-400'
                                      }`}
                                    >
                                      {kf.contrib >= 0 ? '+' : ''}{kf.contrib.toFixed(3)}
                                    </span>
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                        {recent.length > 0 ? (
                          <div className="mt-6 pt-6 border-t border-border/60 w-full">
                            <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3">
                              Recent tournament history (merged PGA / LIV / ESPN rows)
                            </h4>
                            <div className="overflow-x-auto rounded-lg border border-border/60">
                              <table className="w-full text-xs">
                                <thead>
                                  <tr className="bg-secondary/40 text-muted-foreground text-left">
                                    <th className="px-3 py-2 font-medium whitespace-nowrap">Start</th>
                                    <th className="px-3 py-2 font-medium">Event</th>
                                    <th className="px-3 py-2 font-medium">Pos</th>
                                    <th className="px-3 py-2 font-medium">To par</th>
                                    <th className="px-3 py-2 font-medium text-right">R1</th>
                                    <th className="px-3 py-2 font-medium text-right">R2</th>
                                    <th className="px-3 py-2 font-medium text-right">R3</th>
                                    <th className="px-3 py-2 font-medium text-right">R4</th>
                                    <th className="px-3 py-2 font-medium text-right">Tot</th>
                                  </tr>
                                </thead>
                                <tbody>
                                  {recent.map((row, ri) => (
                                    <tr key={`${row.start}-${ri}`} className="border-t border-border/40">
                                      <td className="px-3 py-2 whitespace-nowrap font-mono">{row.start}</td>
                                      <td className="px-3 py-2 max-w-[220px] truncate" title={row.tournament}>
                                        {row.tournament}
                                      </td>
                                      <td className="px-3 py-2">{row.position}</td>
                                      <td className="px-3 py-2 font-mono">{row.scoreToPar || '—'}</td>
                                      <td className="px-3 py-2 text-right font-mono text-muted-foreground">
                                        {row.r1 ?? '—'}
                                      </td>
                                      <td className="px-3 py-2 text-right font-mono text-muted-foreground">
                                        {row.r2 ?? '—'}
                                      </td>
                                      <td className="px-3 py-2 text-right font-mono text-muted-foreground">
                                        {row.r3 ?? '—'}
                                      </td>
                                      <td className="px-3 py-2 text-right font-mono text-muted-foreground">
                                        {row.r4 ?? '—'}
                                      </td>
                                      <td className="px-3 py-2 text-right font-mono">{row.total ?? '—'}</td>
                                    </tr>
                                  ))}
                                </tbody>
                              </table>
                            </div>
                          </div>
                        ) : (
                          <div className="mt-6 pt-6 border-t border-dashed border-border/50 text-xs text-muted-foreground w-full">
                            No recent rows in merged results TSVs for this exact name — check spelling vs archive
                            / ESPN supplement.
                          </div>
                        )}
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
