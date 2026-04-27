'use client';

import { Fragment, useCallback, useEffect, useMemo, useState } from 'react';
import {
  OddsEdgePanel,
  type EdgeEntry,
  type MarketOddsData,
} from '@/components/OddsEdgePanel';

type PredRow = {
  player: string;
  exp_sg_per_round: number;
  sim_win_pct: number;
  sim_top5_pct: number;
  sim_top10_pct: number;
  sim_top20_pct: number;
  best_calibrated_target_made_cut_prob?: number;
  best_calibrated_target_top10_prob?: number;
  best_calibrated_target_top20_prob?: number;
  best_calibrated_target_win_prob?: number;
  best_calibrated_target_made_cut_model?: string;
  best_calibrated_target_top10_model?: string;
  best_calibrated_target_top20_model?: string;
  best_calibrated_target_win_model?: string;
  lr_target_made_cut_prob?: number;
  lr_target_top10_prob?: number;
  lr_target_top20_prob?: number;
  lr_target_win_prob?: number;
  market_implied_win?: number | null;
  best_price_win?: number | null;
  best_book_win?: string | null;
  edge_win?: number | null;
  ev_win?: number | null;
  kelly_win?: number | null;
  book_odds_win?: Record<string, { price: number; implied: number; decimal: number }>;
};

type FormRow = {
  tournament: string;
  start: string;
  position: string;
  scoreToPar: string;
  r1: number | null;
  r2: number | null;
  r3: number | null;
  r4: number | null;
  total: number | null;
};

type TournRow = { tournament: string; start: string; players: number };

type LeaderboardPlayer = {
  player: string;
  toPar: string;
  thru: string;
  totalStrokes: number | null;
  rounds: Record<string, number>;
  status: string;
  position: number;
  positionDisplay: string;
};

type LiveLeaderboard = {
  event: string;
  eventDate: string;
  currentRound: number;
  status: string;
  isCompleted: boolean;
  fetchedAt: string;
  players: LeaderboardPlayer[];
};

type MidTournamentPred = {
  current_pos: number;
  current_pos_display: string;
  player: string;
  pred_name: string;
  to_par: number;
  to_par_display: string;
  r1: number | null;
  r2: number | null;
  total_strokes: number;
  actual_sg_per_round: number;
  pre_sg_per_round: number;
  updated_sg_per_round: number;
  sim_win_pct: number;
  sim_top5_pct: number;
  sim_top10_pct: number;
  sim_top20_pct: number;
  pre_win_prob: number | null;
  pre_top5_pct: number | null;
  pre_top10_prob: number | null;
  pre_top20_prob: number | null;
  pre_rank: number | null;
  rank_change: number | null;
};

type MidTournamentData = {
  meta: {
    type: string;
    event: string;
    status: string;
    rounds_completed: number;
    remaining_rounds: number;
    cut_line: string;
    made_cut: number;
    missed_cut: number;
    n_sims: number;
    actual_weight: number;
    pretournament_weight: number;
    generated_at: string;
    [key: string]: unknown;
  };
  predictions: MidTournamentPred[];
};

type Dashboard = {
  generatedAt: string;
  predictions: PredRow[];
  predictionMeta: Record<string, unknown>;
  espnSupplement: { path: string; rows: number; seasons?: number[] };
  mergedResults?: { mainPath: string; supplementPath: string; mergedRows: number };
  tournaments2026: TournRow[];
  recentByPlayer: Record<string, FormRow[]>;
  marketOdds?: MarketOddsData;
  edges?: EdgeEntry[];
  placementMarkets?: Record<string, any>;
  liveLeaderboard?: LiveLeaderboard;
  midtournament?: MidTournamentData;
};

type Tab = 'leaderboard' | 'predictions' | 'schedule' | 'form' | 'odds';
type PredictionSortKey =
  | 'exp_sg_per_round'
  | 'sim_win_pct'
  | 'sim_top10_pct'
  | 'sim_top20_pct'
  | 'best_calibrated_target_made_cut_prob'
  | 'best_calibrated_target_top10_prob'
  | 'best_calibrated_target_top20_prob'
  | 'best_calibrated_target_win_prob'
  | 'edge_win'
  | 'ev_win';

const DATA_URL = '/data/pga_masters_dashboard.json';

function toParColor(tp: string) {
  const s = tp.trim().toUpperCase();
  if (s === 'E') return 'text-foreground';
  if (s.startsWith('-')) return 'text-red-400';
  if (s.startsWith('+')) {
    const n = parseInt(s.replace('+', ''), 10);
    if (n >= 5) return 'text-blue-400/60';
    return 'text-blue-400';
  }
  return 'text-muted-foreground';
}

function LiveLeaderboardTab({
  leaderboard,
  predictions,
  recentByPlayer,
  midtournament,
}: {
  leaderboard?: LiveLeaderboard;
  predictions: PredRow[];
  recentByPlayer: Record<string, FormRow[]>;
  midtournament?: MidTournamentData;
}) {
  const [expandedPlayer, setExpandedPlayer] = useState<string | null>(null);

  if (!leaderboard || !leaderboard.players.length) {
    return (
      <div className="rounded-lg border border-dashed border-border px-4 py-8 text-center text-muted-foreground text-sm">
        <p className="font-medium mb-2">No live leaderboard available</p>
        <p className="text-xs">
          Re-run{' '}
          <code className="bg-secondary px-1 rounded">python scripts/export_pga_dashboard.py</code>{' '}
          to fetch the current ESPN scoreboard.
        </p>
      </div>
    );
  }

  const predIdx = Object.fromEntries(predictions.map((p) => [p.player, p]));
  const mtIdx = midtournament
    ? Object.fromEntries(midtournament.predictions.map((p) => [p.player, p]))
    : {};
  const mtMeta = midtournament?.meta;
  const hasMT = midtournament != null && midtournament.predictions.length > 0;
  const cutPlayerSet = hasMT ? new Set(midtournament.predictions.map((p) => p.player)) : null;

  const lb = leaderboard;
  const maxRound = lb.currentRound;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <span className="inline-flex items-center gap-2 text-sm font-medium">
          <span className={`h-2.5 w-2.5 rounded-full ${lb.isCompleted ? 'bg-muted-foreground' : 'bg-emerald-500 animate-pulse'}`} />
          {lb.event} — Round {maxRound}
        </span>
        <span className="text-xs text-muted-foreground">
          {lb.status} · {lb.players.length} players ·{' '}
          fetched {new Date(lb.fetchedAt).toLocaleTimeString()}
        </span>
      </div>

      {hasMT && mtMeta && (
        <div className="flex flex-wrap gap-3">
          <div className="rounded-lg border border-emerald-500/30 bg-emerald-500/5 px-3 py-2 text-xs">
            <span className="text-emerald-400 font-bold">Mid-tournament update</span>
            <span className="text-muted-foreground ml-2">
              {mtMeta.made_cut} made cut · Cut line: {mtMeta.cut_line} · {mtMeta.remaining_rounds} rounds remaining ·{' '}
              {mtMeta.n_sims.toLocaleString()} sims · {(mtMeta.actual_weight * 100).toFixed(0)}% actual / {(mtMeta.pretournament_weight * 100).toFixed(0)}% model blend
            </span>
          </div>
        </div>
      )}

      <div className="rounded-xl border border-border bg-card overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border bg-secondary/30">
                <th className="text-left px-3 py-2 font-medium text-muted-foreground w-12">Pos</th>
                <th className="text-left px-3 py-2 font-medium text-muted-foreground">Player</th>
                <th className="text-right px-3 py-2 font-medium text-muted-foreground">To Par</th>
                {Array.from({ length: maxRound }, (_, i) => (
                  <th key={i} className="text-right px-3 py-2 font-medium text-muted-foreground">
                    R{i + 1}
                  </th>
                ))}
                <th className="text-right px-3 py-2 font-medium text-muted-foreground">Tot</th>
                <th className="text-right px-3 py-2 font-medium text-muted-foreground hidden md:table-cell">
                  {hasMT ? 'Upd SG/R' : 'Exp SG/R'}
                </th>
                <th className="text-right px-3 py-2 font-medium text-muted-foreground hidden md:table-cell">Win%</th>
                <th className="text-right px-3 py-2 font-medium text-muted-foreground hidden lg:table-cell">Top 5%</th>
                <th className="text-right px-3 py-2 font-medium text-muted-foreground hidden lg:table-cell">Top 10%</th>
                <th className="text-right px-3 py-2 font-medium text-muted-foreground hidden lg:table-cell">Top 20%</th>
              </tr>
            </thead>
            <tbody>
              {lb.players.map((p) => {
                const pred = predIdx[p.player];
                const mt = mtIdx[p.player];
                const isExpanded = expandedPlayer === p.player;
                const hist = recentByPlayer[p.player] ?? [];
                const isCut = cutPlayerSet != null && !cutPlayerSet.has(p.player);

                const sgVal = mt ? mt.updated_sg_per_round : pred?.exp_sg_per_round;
                const winPct = mt ? mt.sim_win_pct : pred?.sim_win_pct;
                const t5Pct = mt ? mt.sim_top5_pct : pred?.sim_top5_pct;
                const t10Pct = mt ? mt.sim_top10_pct : pred?.sim_top10_pct;
                const t20Pct = mt ? mt.sim_top20_pct : pred?.sim_top20_pct;

                return (
                  <Fragment key={p.player}>
                    <tr
                      className={`border-b border-border/50 hover:bg-secondary/10 cursor-pointer ${
                        isCut ? 'opacity-40' : p.position <= 5 ? 'bg-emerald-500/5' : ''
                      }`}
                      onClick={() => setExpandedPlayer(isExpanded ? null : p.player)}
                    >
                      <td className="px-3 py-2 font-mono text-muted-foreground font-medium">{p.positionDisplay}</td>
                      <td className="px-3 py-2 font-medium">
                        <div className="flex items-center gap-2">
                          {p.player}
                          {isCut && (
                            <span className="text-[10px] bg-red-500/20 text-red-400 px-1.5 py-0.5 rounded font-bold">
                              CUT
                            </span>
                          )}
                          {!isCut && mt && mt.rank_change != null && Math.abs(mt.rank_change) >= 10 && (
                            <span className={`text-[10px] px-1.5 py-0.5 rounded font-bold ${
                              mt.rank_change > 0
                                ? 'bg-emerald-500/20 text-emerald-400'
                                : 'bg-red-500/20 text-red-400'
                            }`}>
                              {mt.rank_change > 0 ? '↑' : '↓'}{Math.abs(mt.rank_change)}
                            </span>
                          )}
                          {!isCut && !mt && pred && pred.exp_sg_per_round > 0.5 && (
                            <span className="text-[10px] bg-emerald-500/20 text-emerald-400 px-1.5 py-0.5 rounded font-bold">
                              MODEL FAV
                            </span>
                          )}
                          {!pred && !mt && (
                            <span className="text-[10px] bg-secondary text-muted-foreground px-1.5 py-0.5 rounded">
                              NO MODEL
                            </span>
                          )}
                        </div>
                      </td>
                      <td className={`px-3 py-2 text-right font-mono font-bold text-lg ${toParColor(p.toPar)}`}>
                        {p.toPar}
                      </td>
                      {Array.from({ length: maxRound }, (_, i) => {
                        const rnd = p.rounds[String(i + 1)];
                        return (
                          <td key={i} className="px-3 py-2 text-right font-mono text-muted-foreground">
                            {rnd ?? '—'}
                          </td>
                        );
                      })}
                      <td className="px-3 py-2 text-right font-mono">{p.totalStrokes ?? '—'}</td>
                      <td className="px-3 py-2 text-right font-mono hidden md:table-cell">
                        {sgVal != null ? (
                          <span className={sgVal >= 0 ? 'text-emerald-400' : 'text-red-400'}>
                            {sgVal >= 0 ? '+' : ''}{sgVal.toFixed(2)}
                          </span>
                        ) : '—'}
                      </td>
                      <td className="px-3 py-2 text-right font-mono hidden md:table-cell">
                        {winPct != null ? (
                          <span className={winPct > 5 ? 'text-emerald-400 font-semibold' : 'text-muted-foreground'}>
                            {winPct.toFixed(1)}%
                          </span>
                        ) : '—'}
                      </td>
                      <td className="px-3 py-2 text-right font-mono hidden lg:table-cell">
                        {t5Pct != null ? `${t5Pct.toFixed(1)}%` : '—'}
                      </td>
                      <td className="px-3 py-2 text-right font-mono hidden lg:table-cell">
                        {t10Pct != null ? `${t10Pct.toFixed(1)}%` : '—'}
                      </td>
                      <td className="px-3 py-2 text-right font-mono hidden lg:table-cell">
                        {t20Pct != null ? `${t20Pct.toFixed(1)}%` : '—'}
                      </td>
                    </tr>
                    {isExpanded && (
                      <tr className="bg-secondary/20 border-b border-border/50">
                        <td colSpan={8 + maxRound} className="px-4 py-3">
                          {mt ? (
                            <div className="space-y-4">
                              <div>
                                <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">
                                  Mid-tournament predictions (after R{mtMeta?.rounds_completed})
                                </div>
                                <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
                                  {([
                                    ['Win', mt.sim_win_pct],
                                    ['Top 5', mt.sim_top5_pct],
                                    ['Top 10', mt.sim_top10_pct],
                                    ['Top 20', mt.sim_top20_pct],
                                    ['Upd SG/R', null, mt.updated_sg_per_round],
                                  ] as [string, number | null, number?][]).map(([label, pct, raw]) => (
                                    <div key={label} className="rounded-md border border-border/50 bg-card/60 px-3 py-2">
                                      <div className="text-[10px] uppercase tracking-wider text-muted-foreground">{label}</div>
                                      <div className="text-sm font-mono font-semibold">
                                        {raw != null ? `${raw >= 0 ? '+' : ''}${raw.toFixed(2)}` : pct != null ? `${pct.toFixed(1)}%` : '—'}
                                      </div>
                                    </div>
                                  ))}
                                </div>
                              </div>
                              {pred && (
                                <div>
                                  <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">
                                    Pre-tournament model (for comparison)
                                  </div>
                                  <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
                                    {([
                                      ['Win', pred.best_calibrated_target_win_prob ?? pred.lr_target_win_prob],
                                      ['Top 10', pred.best_calibrated_target_top10_prob ?? pred.lr_target_top10_prob],
                                      ['Top 20', pred.best_calibrated_target_top20_prob ?? pred.lr_target_top20_prob],
                                      ['Make Cut', pred.best_calibrated_target_made_cut_prob ?? pred.lr_target_made_cut_prob],
                                      ['Pre SG/R', null, pred.exp_sg_per_round],
                                    ] as [string, number | undefined | null, number?][]).map(([label, prob, raw]) => (
                                      <div key={label} className="rounded-md border border-border/50 bg-card/60 px-3 py-2">
                                        <div className="text-[10px] uppercase tracking-wider text-muted-foreground">{label}</div>
                                        <div className="text-sm font-mono font-semibold text-muted-foreground">
                                          {raw != null ? `${raw >= 0 ? '+' : ''}${raw.toFixed(3)}` : prob != null ? `${(prob * 100).toFixed(1)}%` : '—'}
                                        </div>
                                      </div>
                                    ))}
                                  </div>
                                </div>
                              )}
                            </div>
                          ) : pred ? (
                            <div className="space-y-4">
                              <div>
                                <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">
                                  Model probabilities (pre-tournament)
                                </div>
                                <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
                                  {([
                                    ['Win', pred.best_calibrated_target_win_prob ?? pred.lr_target_win_prob],
                                    ['Top 10', pred.best_calibrated_target_top10_prob ?? pred.lr_target_top10_prob],
                                    ['Top 20', pred.best_calibrated_target_top20_prob ?? pred.lr_target_top20_prob],
                                    ['Make Cut', pred.best_calibrated_target_made_cut_prob ?? pred.lr_target_made_cut_prob],
                                    ['MC Win', pred.sim_win_pct / 100],
                                  ] as [string, number | undefined][]).map(([label, prob]) => (
                                    <div key={label} className="rounded-md border border-border/50 bg-card/60 px-3 py-2">
                                      <div className="text-[10px] uppercase tracking-wider text-muted-foreground">{label}</div>
                                      <div className="text-sm font-mono font-semibold">
                                        {prob != null ? `${(prob * 100).toFixed(1)}%` : '—'}
                                      </div>
                                    </div>
                                  ))}
                                </div>
                              </div>
                              {pred.edge_win != null && (
                                <div>
                                  <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">
                                    Market edges
                                  </div>
                                  <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                                    {([
                                      ['Win', pred.edge_win, pred.ev_win, pred.best_price_win],
                                    ] as [string, number | null | undefined, number | null | undefined, number | null | undefined][])
                                      .filter(([, e]) => e != null)
                                      .map(([label, edgeVal, evVal, price]) => (
                                        <div
                                          key={label}
                                          className={`rounded-md border px-3 py-2 ${
                                            edgeVal! > 0.02
                                              ? 'border-emerald-500/30 bg-emerald-500/10'
                                              : 'border-border/50 bg-card/60'
                                          }`}
                                        >
                                          <div className="text-[10px] uppercase tracking-wider text-muted-foreground">{label}</div>
                                          <div className={`text-sm font-mono font-semibold ${edgeVal! > 0.02 ? 'text-emerald-400' : edgeVal! < -0.02 ? 'text-red-400' : ''}`}>
                                            {edgeVal! > 0 ? '+' : ''}{((edgeVal ?? 0) * 100).toFixed(1)}% edge
                                          </div>
                                          <div className="text-[10px] text-muted-foreground">
                                            E[V] {evVal! > 0 ? '+' : ''}{(evVal ?? 0).toFixed(2)}
                                            {price != null && ` · ${price > 0 ? '+' : ''}${price}`}
                                          </div>
                                        </div>
                                      ))}
                                  </div>
                                </div>
                              )}
                            </div>
                          ) : (
                            <p className="text-sm text-muted-foreground">
                              No model predictions available for {p.player}.
                            </p>
                          )}
                          {hist.length > 0 && (
                            <div className="mt-4">
                              <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">
                                Recent tournaments
                              </div>
                              <div className="overflow-x-auto rounded-md border border-border/60 max-w-full">
                                <table className="w-full text-xs">
                                  <thead>
                                    <tr className="text-left text-muted-foreground border-b border-border/50">
                                      <th className="py-1.5 pr-3">Start</th>
                                      <th className="py-1.5 pr-3">Event</th>
                                      <th className="py-1.5 pr-3">Pos</th>
                                      <th className="py-1.5 pr-3">To par</th>
                                      <th className="py-1.5 text-right">Tot</th>
                                    </tr>
                                  </thead>
                                  <tbody>
                                    {hist.slice(0, 6).map((row, hi) => (
                                      <tr key={`${row.start}-${hi}`} className="border-t border-border/40">
                                        <td className="py-1.5 pr-3 font-mono whitespace-nowrap">{row.start}</td>
                                        <td className="py-1.5 pr-3 max-w-[240px] truncate" title={row.tournament}>
                                          {row.tournament}
                                        </td>
                                        <td className="py-1.5 pr-3">{row.position}</td>
                                        <td className="py-1.5 pr-3 font-mono">{row.scoreToPar || '—'}</td>
                                        <td className="py-1.5 text-right font-mono">{row.total ?? '—'}</td>
                                      </tr>
                                    ))}
                                  </tbody>
                                </table>
                              </div>
                            </div>
                          )}
                        </td>
                      </tr>
                    )}
                  </Fragment>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function pctProb(x: number | undefined) {
  if (x === undefined || Number.isNaN(x)) return '—';
  return `${(x * 100).toFixed(1)}%`;
}

export default function PGAPage() {
  const [data, setData] = useState<Dashboard | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>('leaderboard');
  const [sortKey, setSortKey] = useState<PredictionSortKey>('best_calibrated_target_win_prob');
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc');
  const [playerPick, setPlayerPick] = useState<string>('');
  const [expandedPredPlayer, setExpandedPredPlayer] = useState<string | null>(null);

  useEffect(() => {
    fetch(DATA_URL)
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status}`);
        return r.json();
      })
      .then((j: Dashboard) => {
        setData(j);
        setPlayerPick((prev) => prev || j.predictions?.[0]?.player || '');
      })
      .catch(() => setErr(`Missing ${DATA_URL}. Run: cd data-core && python scripts/export_pga_dashboard.py`));
  }, []);

  const sortedPreds = useMemo(() => {
    if (!data?.predictions) return [];
    const rows = [...data.predictions];
    rows.sort((a, b) => {
      const av = a[sortKey] as number;
      const bv = b[sortKey] as number;
      const cmp = (av ?? 0) - (bv ?? 0);
      return sortDir === 'desc' ? -cmp : cmp;
    });
    return rows;
  }, [data, sortKey, sortDir]);

  const toggleSort = useCallback(
    (k: PredictionSortKey) => {
      if (sortKey === k) setSortDir((d) => (d === 'desc' ? 'asc' : 'desc'));
      else {
        setSortKey(k);
        setSortDir('desc');
      }
    },
    [sortKey]
  );

  const hasOdds = data?.edges != null && data.edges.length > 0;

  const formRows = data?.recentByPlayer[playerPick] ?? [];
  const marketSortButtons: { key: PredictionSortKey; label: string }[] = [
    { key: 'best_calibrated_target_win_prob', label: 'Win%' },
    { key: 'best_calibrated_target_top10_prob', label: 'Top 10%' },
    { key: 'best_calibrated_target_top20_prob', label: 'Top 20%' },
    { key: 'best_calibrated_target_made_cut_prob', label: 'Make Cut%' },
    { key: 'exp_sg_per_round', label: 'Exp SG/R' },
    { key: 'sim_win_pct', label: 'MC Win%' },
    { key: 'sim_top10_pct', label: 'MC Top10%' },
    { key: 'sim_top20_pct', label: 'MC Top20%' },
  ];
  const calibOrLR = (row: PredRow, bestKey: keyof PredRow, lrKey: keyof PredRow) => {
    const v = row[bestKey] as number | undefined;
    if (typeof v === 'number') return v;
    return row[lrKey] as number | undefined;
  };
  const modelLabel = (row: PredRow, bestModelKey: keyof PredRow, fallback: string) => {
    const v = row[bestModelKey] as string | undefined;
    return v || fallback;
  };

  return (
    <div className="container mx-auto px-4 py-8 max-w-[1400px]">
      <div className="mb-8">
        <h1 className="text-3xl font-bold tracking-tight mb-2">PGA — Masters model</h1>
        <p className="text-muted-foreground max-w-3xl text-sm leading-relaxed">
          Pre-event Monte Carlo and logistic targets from the v2 feature store.{' '}
          <strong className="text-foreground">ESPN 2026</strong> results are merged via{' '}
          <code className="text-xs bg-secondary px-1 rounded">pga_results_espn_supplement.tsv</code> — refresh with{' '}
          <code className="text-xs bg-secondary px-1 rounded">python scripts/fetch_espn_pga_results.py</code>, then{' '}
          <code className="text-xs bg-secondary px-1 rounded">python -m src.data.build_pga_feature_store</code>, retrain,{' '}
          <code className="text-xs bg-secondary px-1 rounded">predict_masters_tournament.py</code>, and{' '}
          <code className="text-xs bg-secondary px-1 rounded">export_pga_dashboard.py</code>.
        </p>
        {data && (
          <p className="text-xs text-muted-foreground mt-3">
            JSON generated {new Date(data.generatedAt).toLocaleString()} · ESPN supplement rows:{' '}
            {data.espnSupplement?.rows ?? 0}
            {data.mergedResults?.mergedRows != null && (
              <> · Merged result rows: {data.mergedResults.mergedRows}</>
            )}
            {data.predictionMeta?.latest_result_start != null && (
              <> · Latest result start: {String(data.predictionMeta.latest_result_start)}</>
            )}
          </p>
        )}
      </div>

      {err && (
        <div className="rounded-lg border border-amber-500/40 bg-amber-500/10 px-4 py-3 text-sm text-amber-200 mb-6">
          {err}
        </div>
      )}

      <div className="flex gap-2 mb-6 border-b border-border pb-2">
        {(
          [
            ['leaderboard', 'Live Leaderboard'],
            ['predictions', 'Predictions'],
            ['odds', 'Odds & Edges'],
            ['schedule', '2026 ESPN events'],
            ['form', 'Recent form'],
          ] as const
        ).map(([id, label]) => (
          <button
            key={id}
            type="button"
            onClick={() => setTab(id)}
            className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
              tab === id ? 'bg-primary text-primary-foreground' : 'text-muted-foreground hover:text-foreground'
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      {tab === 'leaderboard' && data && (
        <LiveLeaderboardTab
          leaderboard={data.liveLeaderboard}
          predictions={data.predictions}
          recentByPlayer={data.recentByPlayer}
          midtournament={data.midtournament}
        />
      )}

      {tab === 'predictions' && data && (
        <div className="rounded-xl border border-border bg-card overflow-hidden">
          <div className="px-3 py-2 border-b border-border bg-secondary/20 flex flex-wrap items-center gap-2">
            <span className="text-xs text-muted-foreground mr-1">Sort market:</span>
            {marketSortButtons.map((b) => (
              <button
                key={b.key}
                type="button"
                onClick={() => {
                  setSortKey(b.key);
                  setSortDir('desc');
                }}
                className={`text-xs rounded px-2.5 py-1 border transition-colors ${
                  sortKey === b.key
                    ? 'bg-primary text-primary-foreground border-primary'
                    : 'border-border text-muted-foreground hover:text-foreground'
                }`}
              >
                {b.label}
              </button>
            ))}
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border bg-secondary/30">
                  <th className="text-left px-3 py-2 font-medium text-muted-foreground">#</th>
                  <th className="text-left px-3 py-2 font-medium text-muted-foreground">Player</th>
                  {(
                    [
                      ['exp_sg_per_round', 'Exp SG/R'],
                      ['best_calibrated_target_win_prob', 'Win%'],
                      ...(hasOdds ? [['edge_win' as const, 'Edge'], ['ev_win' as const, 'E[V]']] : []),
                      ['best_calibrated_target_top10_prob', 'Top 10%'],
                      ['best_calibrated_target_top20_prob', 'Top 20%'],
                      ['best_calibrated_target_made_cut_prob', 'Cut%'],
                      ['sim_win_pct', 'MC Win%'],
                      ['sim_top10_pct', 'MC T10%'],
                      ['sim_top20_pct', 'MC T20%'],
                    ] as [string, string][]
                  ).map(([k, lab]) => (
                    <th
                      key={k}
                      className="text-right px-3 py-2 font-medium text-muted-foreground cursor-pointer hover:text-foreground whitespace-nowrap"
                      onClick={() => toggleSort(k as PredictionSortKey)}
                    >
                      {lab}
                      {sortKey === k ? (sortDir === 'desc' ? ' ▼' : ' ▲') : ''}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {sortedPreds.map((r, i) => {
                  const open = expandedPredPlayer === r.player;
                  const hist = data.recentByPlayer[r.player] ?? [];
                  return (
                    <Fragment key={r.player}>
                      <tr
                        className="border-b border-border/50 hover:bg-secondary/10 cursor-pointer"
                        onClick={() => setExpandedPredPlayer(open ? null : r.player)}
                      >
                        <td className="px-3 py-2 text-muted-foreground">{i + 1}</td>
                        <td className="px-3 py-2 font-medium">
                          {r.player}
                          <span className="ml-2 text-[10px] text-muted-foreground">
                            {open ? '▼' : '▶'} history
                          </span>
                        </td>
                        <td className="px-3 py-2 text-right font-mono text-emerald-400/90">
                          {r.exp_sg_per_round >= 0 ? '+' : ''}
                          {r.exp_sg_per_round.toFixed(3)}
                        </td>
                        <td className="px-3 py-2 text-right font-semibold">
                          {pctProb(calibOrLR(r, 'best_calibrated_target_win_prob', 'lr_target_win_prob'))}
                        </td>
                        {hasOdds && (
                          <>
                            <td className="px-3 py-2 text-right font-mono">
                              {r.edge_win != null ? (
                                <span className={r.edge_win > 0.02 ? 'text-emerald-400 font-bold' : r.edge_win < -0.02 ? 'text-red-400' : 'text-muted-foreground'}>
                                  {r.edge_win > 0 ? '+' : ''}{(r.edge_win * 100).toFixed(1)}%
                                </span>
                              ) : '—'}
                            </td>
                            <td className="px-3 py-2 text-right font-mono">
                              {r.ev_win != null ? (
                                <span className={r.ev_win > 0 ? 'text-emerald-400 font-semibold' : 'text-muted-foreground'}>
                                  {r.ev_win > 0 ? '+' : ''}{r.ev_win.toFixed(2)}
                                </span>
                              ) : '—'}
                            </td>
                          </>
                        )}
                        <td className="px-3 py-2 text-right">
                          {pctProb(calibOrLR(r, 'best_calibrated_target_top10_prob', 'lr_target_top10_prob'))}
                        </td>
                        <td className="px-3 py-2 text-right">
                          {pctProb(calibOrLR(r, 'best_calibrated_target_top20_prob', 'lr_target_top20_prob'))}
                        </td>
                        <td className="px-3 py-2 text-right">
                          {pctProb(calibOrLR(r, 'best_calibrated_target_made_cut_prob', 'lr_target_made_cut_prob'))}
                        </td>
                        <td className="px-3 py-2 text-right text-muted-foreground">{r.sim_win_pct.toFixed(1)}%</td>
                        <td className="px-3 py-2 text-right text-muted-foreground">{r.sim_top10_pct.toFixed(1)}%</td>
                        <td className="px-3 py-2 text-right text-muted-foreground">{r.sim_top20_pct.toFixed(1)}%</td>
                      </tr>
                      {open && (
                        <tr className="bg-secondary/20 border-b border-border/50">
                          <td colSpan={hasOdds ? 12 : 10} className="px-4 py-3">
                            <div className="mb-3">
                              <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">
                                Best calibrated probabilities by market
                              </div>
                              <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                                {(
                                  [
                                    [
                                      'Make Cut',
                                      calibOrLR(r, 'best_calibrated_target_made_cut_prob', 'lr_target_made_cut_prob'),
                                      modelLabel(r, 'best_calibrated_target_made_cut_model', 'lr'),
                                    ],
                                    [
                                      'Top 10',
                                      calibOrLR(r, 'best_calibrated_target_top10_prob', 'lr_target_top10_prob'),
                                      modelLabel(r, 'best_calibrated_target_top10_model', 'lr'),
                                    ],
                                    [
                                      'Top 20',
                                      calibOrLR(r, 'best_calibrated_target_top20_prob', 'lr_target_top20_prob'),
                                      modelLabel(r, 'best_calibrated_target_top20_model', 'lr'),
                                    ],
                                    [
                                      'Win',
                                      calibOrLR(r, 'best_calibrated_target_win_prob', 'lr_target_win_prob'),
                                      modelLabel(r, 'best_calibrated_target_win_model', 'lr'),
                                    ],
                                  ] as [string, number | undefined, string][]
                                ).map(([label, prob, model]) => (
                                  <div key={label} className="rounded-md border border-border/50 bg-card/60 px-3 py-2">
                                    <div className="text-[10px] uppercase tracking-wider text-muted-foreground">{label}</div>
                                    <div className="text-sm font-mono font-semibold">{pctProb(prob)}</div>
                                    <div className="text-[10px] text-muted-foreground mt-0.5">model: {model}</div>
                                  </div>
                                ))}
                              </div>
                            </div>
                            <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">
                              Recent tournaments (merged archive + ESPN)
                            </div>
                            {hist.length === 0 ? (
                              <p className="text-sm text-muted-foreground">No rows for this name in merged TSVs.</p>
                            ) : (
                              <div className="overflow-x-auto rounded-md border border-border/60 max-w-full">
                                <table className="w-full text-xs">
                                  <thead>
                                    <tr className="text-left text-muted-foreground border-b border-border/50">
                                      <th className="py-1.5 pr-3">Start</th>
                                      <th className="py-1.5 pr-3">Event</th>
                                      <th className="py-1.5 pr-3">Pos</th>
                                      <th className="py-1.5 pr-3">To par</th>
                                      <th className="py-1.5 text-right">Tot</th>
                                    </tr>
                                  </thead>
                                  <tbody>
                                    {hist.map((row, hi) => (
                                      <tr key={`${row.start}-${hi}`} className="border-t border-border/40">
                                        <td className="py-1.5 pr-3 font-mono whitespace-nowrap">{row.start}</td>
                                        <td className="py-1.5 pr-3 max-w-[240px] truncate" title={row.tournament}>
                                          {row.tournament}
                                        </td>
                                        <td className="py-1.5 pr-3">{row.position}</td>
                                        <td className="py-1.5 pr-3 font-mono">{row.scoreToPar || '—'}</td>
                                        <td className="py-1.5 text-right font-mono">{row.total ?? '—'}</td>
                                      </tr>
                                    ))}
                                  </tbody>
                                </table>
                              </div>
                            )}
                          </td>
                        </tr>
                      )}
                    </Fragment>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {tab === 'odds' && data && (
        <div>
          {data.edges && data.edges.length > 0 ? (
            <OddsEdgePanel
              edges={data.edges}
              marketOdds={data.marketOdds ?? { tournament: 'Masters Tournament', fetchedAt: data.generatedAt, commenceTime: '', books: [], overrounds: {}, playerOdds: [] }}
              placementMarkets={data.placementMarkets}
            />
          ) : (
            <div className="rounded-lg border border-dashed border-border px-4 py-8 text-center text-muted-foreground text-sm">
              <p className="font-medium mb-2">No market odds data available</p>
              <p className="text-xs">
                Run{' '}
                <code className="bg-secondary px-1 rounded">python scripts/fetch_pga_odds.py</code>{' '}
                then{' '}
                <code className="bg-secondary px-1 rounded">python scripts/export_pga_dashboard.py</code>{' '}
                to populate odds from The Odds API.
              </p>
            </div>
          )}
        </div>
      )}

      {tab === 'schedule' && data && (
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {data.tournaments2026.map((t) => (
            <div key={`${t.tournament}-${t.start}`} className="rounded-lg border border-border bg-card p-4">
              <div className="font-medium leading-snug">{t.tournament}</div>
              <div className="text-xs text-muted-foreground mt-1">{t.start}</div>
              <div className="text-sm text-muted-foreground mt-2">{t.players} players (ESPN)</div>
            </div>
          ))}
        </div>
      )}

      {tab === 'form' && data && (
        <div className="space-y-4">
          <div className="flex flex-wrap items-center gap-3">
            <label className="text-sm text-muted-foreground">Player</label>
            <select
              className="bg-secondary border border-border rounded-md px-3 py-2 text-sm min-w-[240px]"
              value={playerPick}
              onChange={(e) => setPlayerPick(e.target.value)}
            >
              {data.predictions.map((p) => (
                <option key={p.player} value={p.player}>
                  {p.player}
                </option>
              ))}
            </select>
          </div>
          <div className="rounded-xl border border-border bg-card overflow-hidden">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border bg-secondary/30">
                  <th className="text-left px-3 py-2">Start</th>
                  <th className="text-left px-3 py-2">Event</th>
                  <th className="text-left px-3 py-2">Pos</th>
                  <th className="text-left px-3 py-2">To par</th>
                  <th className="text-right px-3 py-2">R1</th>
                  <th className="text-right px-3 py-2">R2</th>
                  <th className="text-right px-3 py-2">R3</th>
                  <th className="text-right px-3 py-2">R4</th>
                  <th className="text-right px-3 py-2">Total</th>
                </tr>
              </thead>
              <tbody>
                {formRows.length === 0 ? (
                  <tr>
                    <td colSpan={9} className="px-3 py-6 text-center text-muted-foreground">
                      No ESPN rows for this player (name mismatch vs supplement).
                    </td>
                  </tr>
                ) : (
                  formRows.map((row, i) => (
                    <tr key={`${row.start}-${i}`} className="border-b border-border/50">
                      <td className="px-3 py-2 whitespace-nowrap">{row.start}</td>
                      <td className="px-3 py-2 max-w-[280px] truncate" title={row.tournament}>
                        {row.tournament}
                      </td>
                      <td className="px-3 py-2">{row.position}</td>
                      <td className="px-3 py-2 font-mono">{row.scoreToPar}</td>
                      <td className="px-3 py-2 text-right font-mono text-muted-foreground">{row.r1 ?? '—'}</td>
                      <td className="px-3 py-2 text-right font-mono text-muted-foreground">{row.r2 ?? '—'}</td>
                      <td className="px-3 py-2 text-right font-mono text-muted-foreground">{row.r3 ?? '—'}</td>
                      <td className="px-3 py-2 text-right font-mono text-muted-foreground">{row.r4 ?? '—'}</td>
                      <td className="px-3 py-2 text-right font-mono">{row.total ?? '—'}</td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
