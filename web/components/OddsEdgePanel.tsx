'use client';

import React, { useState } from 'react';

export interface EdgeEntry {
  player: string;
  market: string;
  modelProb: number;
  marketImplied: number;
  edge: number;
  ev: number;
  kellyFraction: number;
  bestPrice: number;
  bestBook: string;
  signal: 'positive' | 'negative' | 'neutral';
}

export interface PlayerOddsEntry {
  player: string;
  apiName: string;
  bestPrice: number;
  bestBook: string;
  consensusImplied: number;
  bookOdds: Record<string, { price: number; implied: number; decimal: number }>;
}

export interface MarketOddsData {
  tournament: string;
  fetchedAt: string;
  commenceTime: string;
  books: string[];
  overrounds: Record<string, number>;
  playerOdds: PlayerOddsEntry[];
}

export interface PlacementMarketSummary {
  market: string;
  books: string[];
  overrounds: Record<string, number>;
  capturedAt: string;
  playerOdds: PlayerOddsEntry[];
}

export interface MatchupEntry {
  id: string;
  market: 'matchup';
  book: string;
  capturedAt: string;
  source: string;
  roundNum: number | null;
  notes: string;
  matchup: {
    playerA: string;
    playerB: string;
    oddsA: { price: number; implied: number };
    oddsB: { price: number; implied: number };
  };
}

const MARKET_LABELS: Record<string, string> = {
  win: 'Outright Win',
  top5: 'Top 5',
  top10: 'Top 10',
  top20: 'Top 20',
  made_cut: 'Make Cut',
  frl: 'First Round Leader',
};

function formatAmerican(price: number) {
  return price > 0 ? `+${price}` : `${price}`;
}

function timeSince(isoDate: string): string {
  const diff = Date.now() - new Date(isoDate).getTime();
  const hours = Math.floor(diff / 3600000);
  const mins = Math.floor((diff % 3600000) / 60000);
  if (hours > 0) return `${hours}h ${mins}m ago`;
  return `${mins}m ago`;
}

function ValueBetsTable({ edges }: { edges: EdgeEntry[] }) {
  const [marketFilter, setMarketFilter] = useState<string>('all');

  const markets = Array.from(new Set(edges.map(e => e.market)));
  const filtered = marketFilter === 'all' ? edges : edges.filter(e => e.market === marketFilter);
  const positive = filtered.filter(e => e.signal === 'positive');
  const marginal = filtered.filter(e => e.signal === 'neutral' && e.edge > 0);

  const renderRow = (e: EdgeEntry, i: number) => (
    <tr key={`${e.player}-${e.market}-${i}`} className="border-b border-border/50 hover:bg-secondary/20 transition-colors">
      <td className="px-4 py-3 font-medium">{i + 1}</td>
      <td className="px-4 py-3 font-medium">{e.player}</td>
      <td className="px-4 py-3 text-center">
        <span className="text-[10px] font-bold uppercase tracking-wider bg-secondary px-2 py-0.5 rounded">
          {MARKET_LABELS[e.market] ?? e.market}
        </span>
      </td>
      <td className="px-4 py-3 text-right font-mono">{(e.modelProb * 100).toFixed(1)}%</td>
      <td className="px-4 py-3 text-right font-mono text-muted-foreground">{(e.marketImplied * 100).toFixed(1)}%</td>
      <td className="px-4 py-3 text-right font-mono">
        <span className={e.edge > 0.02 ? 'text-emerald-400 font-bold' : 'text-amber-400'}>
          +{(e.edge * 100).toFixed(1)}%
        </span>
      </td>
      <td className="px-4 py-3 text-right font-mono">
        <span className={e.ev > 0 ? 'text-emerald-400 font-semibold' : 'text-muted-foreground'}>
          {e.ev > 0 ? '+' : ''}{e.ev.toFixed(2)}
        </span>
      </td>
      <td className="px-4 py-3 text-right font-mono text-xs">{(e.kellyFraction * 100).toFixed(2)}%</td>
      <td className="px-4 py-3 text-right font-mono text-xs">
        {formatAmerican(e.bestPrice)}{' '}
        <span className="text-muted-foreground">({e.bestBook})</span>
      </td>
    </tr>
  );

  return (
    <div className="space-y-3">
      {markets.length > 1 && (
        <div className="flex gap-1.5 flex-wrap">
          <button
            onClick={() => setMarketFilter('all')}
            className={`px-3 py-1 text-xs rounded-md border transition-colors ${
              marketFilter === 'all' ? 'bg-primary text-primary-foreground border-primary' : 'border-border text-muted-foreground hover:text-foreground'
            }`}
          >
            All Markets ({edges.filter(e => e.edge > 0).length})
          </button>
          {markets.map(m => {
            const mEdges = edges.filter(e => e.market === m && e.edge > 0);
            return (
              <button
                key={m}
                onClick={() => setMarketFilter(m)}
                className={`px-3 py-1 text-xs rounded-md border transition-colors ${
                  marketFilter === m ? 'bg-primary text-primary-foreground border-primary' : 'border-border text-muted-foreground hover:text-foreground'
                }`}
              >
                {MARKET_LABELS[m] ?? m} ({mEdges.length})
              </button>
            );
          })}
        </div>
      )}
      {positive.length === 0 && marginal.length === 0 ? (
        <div className="text-sm text-muted-foreground py-6 text-center border border-dashed rounded-xl">
          No positive-edge bets in {marketFilter === 'all' ? 'any market' : MARKET_LABELS[marketFilter] ?? marketFilter}.
        </div>
      ) : (
        <div className="overflow-x-auto rounded-xl border border-border">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border bg-secondary/30">
                <th className="text-left px-4 py-3 font-medium text-muted-foreground w-8">#</th>
                <th className="text-left px-4 py-3 font-medium text-muted-foreground">Player</th>
                <th className="text-center px-4 py-3 font-medium text-muted-foreground">Market</th>
                <th className="text-right px-4 py-3 font-medium text-muted-foreground">Model</th>
                <th className="text-right px-4 py-3 font-medium text-muted-foreground">Implied</th>
                <th className="text-right px-4 py-3 font-medium text-muted-foreground">Edge</th>
                <th className="text-right px-4 py-3 font-medium text-muted-foreground">E[V]/$1</th>
                <th className="text-right px-4 py-3 font-medium text-muted-foreground">Kelly ¼</th>
                <th className="text-right px-4 py-3 font-medium text-muted-foreground">Best Price</th>
              </tr>
            </thead>
            <tbody>
              {positive.length > 0 && (
                <>
                  <tr><td colSpan={9} className="px-4 py-2 text-xs font-bold uppercase tracking-wider text-emerald-400 bg-emerald-500/5">Strong Edge (&gt;2%)</td></tr>
                  {positive.map((e, i) => renderRow(e, i))}
                </>
              )}
              {marginal.length > 0 && (
                <>
                  <tr><td colSpan={9} className="px-4 py-2 text-xs font-bold uppercase tracking-wider text-amber-400 bg-amber-500/5">Marginal Edge (0-2%)</td></tr>
                  {marginal.map((e, i) => renderRow(e, positive.length + i))}
                </>
              )}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function CrossBookComparison({ playerOdds, topN = 15 }: { playerOdds: PlayerOddsEntry[]; topN?: number }) {
  const top = playerOdds
    .filter(po => Object.keys(po.bookOdds).length > 1)
    .slice(0, topN);

  if (top.length === 0) return null;

  const allBooks = Array.from(new Set(top.flatMap(po => Object.keys(po.bookOdds)))).sort();

  return (
    <div className="overflow-x-auto rounded-xl border border-border">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border bg-secondary/30">
            <th className="text-left px-4 py-3 font-medium text-muted-foreground">Player</th>
            <th className="text-right px-4 py-3 font-medium text-muted-foreground">Consensus</th>
            {allBooks.map(b => (
              <th key={b} className="text-right px-4 py-3 font-medium text-muted-foreground capitalize">{b}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {top.map(po => {
            const bestBook = po.bestBook;
            return (
              <tr key={po.player} className="border-b border-border/50 hover:bg-secondary/20 transition-colors">
                <td className="px-4 py-2.5 font-medium whitespace-nowrap">{po.player}</td>
                <td className="px-4 py-2.5 text-right font-mono text-muted-foreground">{(po.consensusImplied * 100).toFixed(1)}%</td>
                {allBooks.map(b => {
                  const odds = po.bookOdds[b];
                  if (!odds) return <td key={b} className="px-4 py-2.5 text-right text-muted-foreground/40">—</td>;
                  const isBest = b === bestBook;
                  return (
                    <td key={b} className={`px-4 py-2.5 text-right font-mono text-xs ${isBest ? 'text-emerald-400 font-bold' : ''}`}>
                      {formatAmerican(odds.price)}
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function OverroundBar({ overrounds }: { overrounds: Record<string, number> }) {
  const entries = Object.entries(overrounds).sort(([, a], [, b]) => a - b);
  return (
    <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
      {entries.map(([book, ov]) => {
        const vigPct = ((ov - 1) * 100).toFixed(1);
        const width = Math.min(((ov - 1) / 0.8) * 100, 100);
        const color = ov < 1.2 ? 'bg-emerald-500' : ov < 1.4 ? 'bg-amber-500' : 'bg-red-500';
        return (
          <div key={book} className="rounded-lg border border-border p-3">
            <div className="flex justify-between items-center mb-1.5">
              <span className="text-xs font-medium capitalize">{book}</span>
              <span className="text-xs font-mono text-muted-foreground">{vigPct}% vig</span>
            </div>
            <div className="h-1.5 bg-secondary/50 rounded-full overflow-hidden">
              <div className={`h-full rounded-full ${color}`} style={{ width: `${width}%` }} />
            </div>
          </div>
        );
      })}
    </div>
  );
}

function MarketSummaryCards({
  placementMarkets,
  edges,
}: {
  placementMarkets: Record<string, PlacementMarketSummary>;
  edges: EdgeEntry[];
}) {
  const marketKeys = Object.keys(placementMarkets).filter(k => k !== 'matchups');
  if (marketKeys.length === 0) return null;

  return (
    <div className="space-y-4">
      <h4 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">
        Placement Market Odds (from screenshots / manual entry)
      </h4>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {marketKeys.map(mkt => {
          const summary = placementMarkets[mkt];
          const mktEdges = edges.filter(e => e.market === mkt);
          const posEdges = mktEdges.filter(e => e.signal === 'positive');
          const bestEdge = posEdges[0];

          return (
            <div key={mkt} className="rounded-xl border border-border bg-card p-4">
              <div className="flex items-center justify-between mb-3">
                <h5 className="font-bold text-sm">{MARKET_LABELS[mkt] ?? mkt}</h5>
                <span className="text-[10px] text-muted-foreground">
                  {summary.books.join(', ')} &middot; {timeSince(summary.capturedAt)}
                </span>
              </div>
              <div className="grid grid-cols-3 gap-2 text-center mb-3">
                <div>
                  <div className="text-[10px] text-muted-foreground uppercase">Players</div>
                  <div className="text-lg font-bold font-mono">{summary.playerOdds.length}</div>
                </div>
                <div>
                  <div className="text-[10px] text-muted-foreground uppercase">Pos. Edges</div>
                  <div className={`text-lg font-bold font-mono ${posEdges.length > 0 ? 'text-emerald-400' : ''}`}>
                    {posEdges.length}
                  </div>
                </div>
                <div>
                  <div className="text-[10px] text-muted-foreground uppercase">Books</div>
                  <div className="text-lg font-bold font-mono">{summary.books.length}</div>
                </div>
              </div>
              {bestEdge && (
                <div className="rounded-lg bg-emerald-500/10 border border-emerald-500/20 p-2.5 text-xs">
                  <div className="flex justify-between items-center">
                    <span className="font-medium">{bestEdge.player}</span>
                    <span className="text-emerald-400 font-bold">+{(bestEdge.edge * 100).toFixed(1)}% edge</span>
                  </div>
                  <div className="flex justify-between items-center mt-1 text-muted-foreground">
                    <span>Model {(bestEdge.modelProb * 100).toFixed(1)}% vs Market {(bestEdge.marketImplied * 100).toFixed(1)}%</span>
                    <span className="font-mono">{formatAmerican(bestEdge.bestPrice)} ({bestEdge.bestBook})</span>
                  </div>
                </div>
              )}
              {posEdges.length === 0 && (
                <div className="text-xs text-muted-foreground text-center py-2">No edges in this market</div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function MatchupsPanel({ matchups }: { matchups: MatchupEntry[] }) {
  if (!matchups || matchups.length === 0) return null;
  return (
    <div className="space-y-3">
      <h4 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">
        Head-to-Head Matchups
      </h4>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        {matchups.map(m => (
          <div key={m.id} className="rounded-xl border border-border bg-card p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-[10px] text-muted-foreground uppercase tracking-wider">{m.book}</span>
              {m.roundNum && <span className="text-[10px] text-muted-foreground">R{m.roundNum}</span>}
            </div>
            <div className="flex items-center justify-between">
              <div className="text-center flex-1">
                <div className="font-medium text-sm">{m.matchup.playerA}</div>
                <div className="font-mono font-bold text-lg mt-1">{formatAmerican(m.matchup.oddsA.price)}</div>
                <div className="text-[10px] text-muted-foreground">{(m.matchup.oddsA.implied * 100).toFixed(1)}%</div>
              </div>
              <div className="text-muted-foreground text-xs font-bold px-3">vs</div>
              <div className="text-center flex-1">
                <div className="font-medium text-sm">{m.matchup.playerB}</div>
                <div className="font-mono font-bold text-lg mt-1">{formatAmerican(m.matchup.oddsB.price)}</div>
                <div className="text-[10px] text-muted-foreground">{(m.matchup.oddsB.implied * 100).toFixed(1)}%</div>
              </div>
            </div>
            {m.notes && <div className="text-[10px] text-muted-foreground mt-2 pt-2 border-t border-border/50">{m.notes}</div>}
          </div>
        ))}
      </div>
    </div>
  );
}

export function OddsEdgePanel({
  edges,
  marketOdds,
  placementMarkets,
}: {
  edges: EdgeEntry[];
  marketOdds: MarketOddsData;
  placementMarkets?: Record<string, any>;
}) {
  const [section, setSection] = useState<'value' | 'comparison' | 'overround' | 'markets' | 'matchups'>('value');

  const staleHours = (Date.now() - new Date(marketOdds.fetchedAt).getTime()) / 3600000;
  const isStale = staleHours > 6;

  const hasPlacementMarkets = placementMarkets && Object.keys(placementMarkets).some(k => k !== 'matchups');
  const hasMatchups = placementMarkets?.matchups && placementMarkets.matchups.length > 0;

  const tabs: [string, string][] = [
    ['value', 'Value Bets'],
    ['comparison', 'Line Shopping'],
    ['overround', 'Vig Analysis'],
  ];
  if (hasPlacementMarkets) tabs.push(['markets', 'Placement Markets']);
  if (hasMatchups) tabs.push(['matchups', 'Matchups']);

  return (
    <div className="space-y-6">
      {isStale && (
        <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 px-4 py-3 text-sm text-amber-400 flex items-center gap-2">
          <svg className="w-4 h-4 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z" />
          </svg>
          Odds data is {timeSince(marketOdds.fetchedAt)} old. Re-run the fetch script for current lines.
        </div>
      )}

      <div className="flex items-center justify-between flex-wrap gap-3">
        <div>
          <h3 className="text-lg font-semibold">{marketOdds.tournament} — Market Odds & Edges</h3>
          <p className="text-xs text-muted-foreground mt-0.5">
            {marketOdds.books.length} books (API) &middot; {marketOdds.playerOdds.length} players &middot; Updated {timeSince(marketOdds.fetchedAt)}
            {hasPlacementMarkets && (
              <> &middot; + placement markets from manual entry</>
            )}
          </p>
        </div>
        <div className="flex gap-1 bg-secondary/50 rounded-lg p-0.5 flex-wrap">
          {tabs.map(([key, label]) => (
            <button
              key={key}
              onClick={() => setSection(key as typeof section)}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors ${
                section === key ? 'bg-card text-foreground shadow-sm' : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      {section === 'value' && <ValueBetsTable edges={edges} />}
      {section === 'comparison' && <CrossBookComparison playerOdds={marketOdds.playerOdds} />}
      {section === 'overround' && <OverroundBar overrounds={marketOdds.overrounds} />}
      {section === 'markets' && placementMarkets && (
        <MarketSummaryCards placementMarkets={placementMarkets} edges={edges} />
      )}
      {section === 'matchups' && placementMarkets?.matchups && (
        <MatchupsPanel matchups={placementMarkets.matchups} />
      )}
    </div>
  );
}
