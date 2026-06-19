'use client';

import { Fragment, useCallback, useEffect, useMemo, useState } from 'react';
import {
  OddsEdgePanel,
  type EdgeEntry,
  type MarketOddsData,
  type PlacementMarkets,
} from '@/components/OddsEdgePanel';

type PredRow = {
  player: string;
  player_id?: string | null;
  exp_sg_per_round: number;
  sim_win_pct: number;
  sim_top5_pct: number;
  sim_top10_pct: number;
  sim_top20_pct: number;
  projected_total_strokes?: number | null;
  projected_score_to_par?: number | null;
  confidence?: number | null;
  quality_flags?: string | string[] | null;
  source?: string | null;
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
  r3?: number | null;
  r4?: number | null;
  total_strokes: number;
  actual_sg_per_round: number;
  pre_sg_per_round: number;
  updated_sg_per_round: number;
  sim_make_cut_pct?: number | null;
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
    cut_line?: string | null;
    made_cut?: number | null;
    missed_cut?: number | null;
    active_players?: number | null;
    cut_after_round?: number | null;
    cut_applied?: boolean | null;
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
  event?: {
    eventKey: string;
    name: string;
    season: number;
    course: string;
    par: number;
    yardage?: number;
    startDate: string;
    endDate: string;
    status: string;
  };
  predictions: PredRow[];
  normalizedMarkets?: Record<string, unknown>[];
  gaps?: string[];
  predictionMeta: Record<string, unknown>;
  espnSupplement: { path: string; rows: number; seasons?: number[] };
  mergedResults?: { mainPath: string; supplementPath: string; mergedRows: number };
  tournaments2026: TournRow[];
  recentByPlayer: Record<string, FormRow[]>;
  marketOdds?: MarketOddsData;
  edges?: EdgeEntry[];
  placementMarkets?: PlacementMarkets;
  liveLeaderboard?: LiveLeaderboard;
  midtournament?: MidTournamentData;
};

type Tab = 'predictions' | 'leaderboard' | 'schedule' | 'form' | 'odds';
type PredictionSortKey =
  | 'win_pct'
  | 'make_cut_pct'
  | 'top5_pct'
  | 'top10_pct'
  | 'top20_pct'
  | 'updated_sg'
  | 'to_par'
  | 'win_delta_pct'
  | 'pre_win_pct';

type OutlookRow = {
  player: string;
  pre?: PredRow;
  mt?: MidTournamentPred;
  rankLabel: string;
  positionLabel?: string;
  toParDisplay?: string;
  toParValue?: number | null;
  completedRounds: (number | null)[];
  totalStrokes?: number | null;
  updatedSg?: number | null;
  preSg?: number | null;
  winPct?: number | null;
  makeCutPct?: number | null;
  top5Pct?: number | null;
  top10Pct?: number | null;
  top20Pct?: number | null;
  preWinPct?: number | null;
  preMakeCutPct?: number | null;
  preTop10Pct?: number | null;
  preTop20Pct?: number | null;
  winDeltaPct?: number | null;
  rankChange?: number | null;
};

const DATA_URL = '/data/pga_tournaments/current.json';

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

function asPct(prob: number | null | undefined) {
  if (prob == null || Number.isNaN(prob)) return null;
  return prob * 100;
}

function pctDisplay(value: number | null | undefined) {
  if (value == null || Number.isNaN(value)) return '—';
  return `${value.toFixed(1)}%`;
}

function signedPctDisplay(value: number | null | undefined) {
  if (value == null || Number.isNaN(value)) return '—';
  return `${value > 0 ? '+' : ''}${value.toFixed(1)} pts`;
}

function strokesDisplay(value: number | null | undefined, digits = 2) {
  if (value == null || Number.isNaN(value)) return '—';
  return `${value >= 0 ? '+' : ''}${value.toFixed(digits)}`;
}

function preWinPct(row: PredRow | undefined) {
  if (!row) return null;
  return asPct(row.best_calibrated_target_win_prob ?? row.lr_target_win_prob) ?? row.sim_win_pct;
}

function preMakeCutPct(row: PredRow | undefined) {
  if (!row) return null;
  return asPct(row.best_calibrated_target_made_cut_prob ?? row.lr_target_made_cut_prob);
}

function preTop10Pct(row: PredRow | undefined) {
  if (!row) return null;
  return asPct(row.best_calibrated_target_top10_prob ?? row.lr_target_top10_prob) ?? row.sim_top10_pct;
}

function preTop20Pct(row: PredRow | undefined) {
  if (!row) return null;
  return asPct(row.best_calibrated_target_top20_prob ?? row.lr_target_top20_prob) ?? row.sim_top20_pct;
}

function buildOutlookRows(data: Dashboard): OutlookRow[] {
  const preByPlayer = Object.fromEntries(data.predictions.map((row) => [row.player, row]));
  const mtRows = data.midtournament?.predictions ?? [];
  if (mtRows.length > 0) {
    return mtRows.map((mt, index) => {
      const pre = preByPlayer[mt.pred_name] ?? preByPlayer[mt.player];
      const winPct = mt.sim_win_pct;
      const preWin = preWinPct(pre);
      return {
        player: mt.pred_name || mt.player,
        pre,
        mt,
        rankLabel: String(index + 1),
        positionLabel: mt.current_pos_display,
        toParDisplay: mt.to_par_display,
        toParValue: mt.to_par,
        completedRounds: [mt.r1, mt.r2, mt.r3 ?? null, mt.r4 ?? null],
        totalStrokes: mt.total_strokes,
        updatedSg: mt.updated_sg_per_round,
        preSg: mt.pre_sg_per_round,
        winPct,
        makeCutPct: mt.sim_make_cut_pct ?? null,
        top5Pct: mt.sim_top5_pct,
        top10Pct: mt.sim_top10_pct,
        top20Pct: mt.sim_top20_pct,
        preWinPct: preWin,
        preMakeCutPct: preMakeCutPct(pre),
        preTop10Pct: preTop10Pct(pre),
        preTop20Pct: preTop20Pct(pre),
        winDeltaPct: preWin == null ? null : winPct - preWin,
        rankChange: mt.rank_change,
      };
    });
  }

  return data.predictions.map((pre, index) => {
    const winPct = preWinPct(pre);
    return {
      player: pre.player,
      pre,
      rankLabel: String(index + 1),
      completedRounds: [],
      updatedSg: pre.exp_sg_per_round,
      preSg: pre.exp_sg_per_round,
      winPct,
      makeCutPct: preMakeCutPct(pre),
      top5Pct: pre.sim_top5_pct,
      top10Pct: preTop10Pct(pre),
      top20Pct: preTop20Pct(pre),
      preWinPct: winPct,
      preMakeCutPct: preMakeCutPct(pre),
      preTop10Pct: preTop10Pct(pre),
      preTop20Pct: preTop20Pct(pre),
      winDeltaPct: null,
    };
  });
}

function valueForSort(row: OutlookRow, key: PredictionSortKey) {
  switch (key) {
    case 'win_pct':
      return row.winPct;
    case 'make_cut_pct':
      return row.makeCutPct;
    case 'top5_pct':
      return row.top5Pct;
    case 'top10_pct':
      return row.top10Pct;
    case 'top20_pct':
      return row.top20Pct;
    case 'updated_sg':
      return row.updatedSg;
    case 'to_par':
      return row.toParValue == null ? null : -row.toParValue;
    case 'win_delta_pct':
      return row.winDeltaPct;
    case 'pre_win_pct':
      return row.preWinPct;
    default:
      return null;
  }
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
        <p className="font-medium mb-2">No score context available</p>
        <p className="text-xs">
          Re-run{' '}
          <code className="bg-secondary px-1 rounded">python scripts/export_pga_tournament_dashboard.py</code>{' '}
          to fetch the current ESPN score snapshot.
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
  const cutContext = mtMeta?.cut_applied
    ? `${mtMeta.made_cut ?? '—'} made cut · Cut line: ${mtMeta.cut_line ?? '—'}`
    : `Cut after Round ${mtMeta?.cut_after_round ?? 2}`;

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
            <span className="text-emerald-400 font-bold">Round outlook update</span>
            <span className="text-muted-foreground ml-2">
              {cutContext} · {mtMeta.remaining_rounds} rounds remaining ·{' '}
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
                                  Round outlook (after R{mtMeta?.rounds_completed})
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

export default function PGAPage() {
  const [data, setData] = useState<Dashboard | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>('predictions');
  const [sortKey, setSortKey] = useState<PredictionSortKey>('win_pct');
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
      .catch(() => setErr(`Missing ${DATA_URL}. Run: cd data-core && python scripts/export_pga_tournament_dashboard.py`));
  }, []);

  const sortedOutlookRows = useMemo(() => {
    if (!data) return [];
    const rows = buildOutlookRows(data);
    rows.sort((a, b) => {
      const av = valueForSort(a, sortKey);
      const bv = valueForSort(b, sortKey);
      const cmp = (av ?? -9999) - (bv ?? -9999);
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

  const eventName = data?.event?.name ?? 'PGA Tournament';
  const eventCourse = data?.event?.course ?? 'Tournament course';
  const eventPar = data?.event?.par;
  const roundMeta = data?.midtournament?.meta;
  const hasRoundUpdate = data?.midtournament != null && data.midtournament.predictions.length > 0;
  const completedRoundLabel = roundMeta ? `After Round ${roundMeta.rounds_completed}` : 'Pre-tournament';
  const liveRoundLabel = data?.liveLeaderboard
    ? `Round ${data.liveLeaderboard.currentRound} ${data.liveLeaderboard.status.toLowerCase()}`
    : data?.event?.status?.replace(/_/g, ' ');

  const formRows = data?.recentByPlayer[playerPick] ?? [];
  const marketSortButtons: { key: PredictionSortKey; label: string }[] = [
    { key: 'win_pct', label: 'Win%' },
    { key: 'make_cut_pct', label: 'Make Cut%' },
    { key: 'top5_pct', label: 'Top 5%' },
    { key: 'top10_pct', label: 'Top 10%' },
    { key: 'top20_pct', label: 'Top 20%' },
    { key: 'updated_sg', label: 'Upd SG/R' },
    { key: 'to_par', label: 'Score' },
    { key: 'win_delta_pct', label: 'Win Δ' },
  ];
  return (
    <div className="container mx-auto px-4 py-8 max-w-[1400px]">
      <div className="mb-8">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div>
            <h1 className="text-3xl font-bold tracking-tight mb-2">PGA Round Outlook</h1>
            <p className="text-muted-foreground max-w-3xl text-sm leading-relaxed">
              {eventName} probabilities for {eventCourse}
              {eventPar ? `, par ${eventPar}` : ''}.
            </p>
          </div>
          <div className="text-right">
            <div className="text-sm font-semibold">{completedRoundLabel}</div>
            {liveRoundLabel && <div className="text-xs text-muted-foreground">{liveRoundLabel}</div>}
          </div>
        </div>
        {hasRoundUpdate && roundMeta && (
          <div className="mt-4 grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
            <div className="rounded-lg border border-border bg-card px-3 py-2">
              <div className="text-[10px] uppercase tracking-wider text-muted-foreground">Simulation</div>
              <div className="text-sm font-semibold">{roundMeta.n_sims.toLocaleString()} runs</div>
            </div>
            <div className="rounded-lg border border-border bg-card px-3 py-2">
              <div className="text-[10px] uppercase tracking-wider text-muted-foreground">Model Blend</div>
              <div className="text-sm font-semibold">
                {(roundMeta.actual_weight * 100).toFixed(0)}% round data / {(roundMeta.pretournament_weight * 100).toFixed(0)}% pre
              </div>
            </div>
            <div className="rounded-lg border border-border bg-card px-3 py-2">
              <div className="text-[10px] uppercase tracking-wider text-muted-foreground">Players Simulated</div>
              <div className="text-sm font-semibold">{String(roundMeta.active_players ?? data?.midtournament?.predictions.length ?? '—')}</div>
            </div>
            <div className="rounded-lg border border-border bg-card px-3 py-2">
              <div className="text-[10px] uppercase tracking-wider text-muted-foreground">Cut State</div>
              <div className="text-sm font-semibold">
                {roundMeta.cut_applied ? `Cut line ${String(roundMeta.cut_line ?? '—')}` : `Cut after Round ${String(roundMeta.cut_after_round ?? 2)}`}
              </div>
            </div>
          </div>
        )}
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
            {data.predictionMeta?.model_version != null && (
              <> · Model: {String(data.predictionMeta.model_version)}</>
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
            ['predictions', 'Round Outlook'],
            ['leaderboard', 'Score Context'],
            ['odds', 'Odds & Edges'],
            ['schedule', '2026 events'],
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
            <span className="text-xs text-muted-foreground mr-1">Sort:</span>
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
                  {hasRoundUpdate && <th className="text-right px-3 py-2 font-medium text-muted-foreground">Pos</th>}
                  {hasRoundUpdate && <th className="text-right px-3 py-2 font-medium text-muted-foreground">Score</th>}
                  {hasRoundUpdate && Array.from({ length: roundMeta?.rounds_completed ?? 0 }, (_, i) => (
                    <th key={`r${i + 1}`} className="text-right px-3 py-2 font-medium text-muted-foreground">
                      R{i + 1}
                    </th>
                  ))}
                  {(
                    [
                      ['updated_sg', hasRoundUpdate ? 'Upd SG/R' : 'Exp SG/R'],
                      ['win_pct', 'Win%'],
                      ['make_cut_pct', 'Cut%'],
                      ['top5_pct', 'Top 5%'],
                      ['top10_pct', 'Top 10%'],
                      ['top20_pct', 'Top 20%'],
                      ['win_delta_pct', 'Win Δ'],
                      ['pre_win_pct', 'Pre Win%'],
                    ] as [PredictionSortKey, string][]
                  ).map(([k, lab]) => (
                    <th
                      key={k}
                      className="text-right px-3 py-2 font-medium text-muted-foreground cursor-pointer hover:text-foreground whitespace-nowrap"
                      onClick={() => toggleSort(k)}
                    >
                      {lab}
                      {sortKey === k ? (sortDir === 'desc' ? ' ▼' : ' ▲') : ''}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {sortedOutlookRows.map((r, i) => {
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
                          <div className="flex items-center gap-2">
                            <span>{r.player}</span>
                            {r.rankChange != null && Math.abs(r.rankChange) >= 10 && (
                              <span className={`text-[10px] px-1.5 py-0.5 rounded font-bold ${
                                r.rankChange > 0
                                  ? 'bg-emerald-500/20 text-emerald-400'
                                  : 'bg-red-500/20 text-red-400'
                              }`}>
                                {r.rankChange > 0 ? '+' : ''}{r.rankChange}
                              </span>
                            )}
                            <span className="text-[10px] text-muted-foreground">{open ? 'Hide' : 'Details'}</span>
                          </div>
                        </td>
                        {hasRoundUpdate && (
                          <td className="px-3 py-2 text-right font-mono text-muted-foreground">{r.positionLabel ?? '—'}</td>
                        )}
                        {hasRoundUpdate && (
                          <td className={`px-3 py-2 text-right font-mono font-semibold ${toParColor(r.toParDisplay ?? '')}`}>
                            {r.toParDisplay ?? '—'}
                          </td>
                        )}
                        {hasRoundUpdate && Array.from({ length: roundMeta?.rounds_completed ?? 0 }, (_, ri) => (
                          <td key={ri} className="px-3 py-2 text-right font-mono text-muted-foreground">
                            {r.completedRounds[ri] ?? '—'}
                          </td>
                        ))}
                        <td className="px-3 py-2 text-right font-mono text-emerald-400/90">
                          {strokesDisplay(r.updatedSg, hasRoundUpdate ? 2 : 3)}
                        </td>
                        <td className="px-3 py-2 text-right font-semibold">{pctDisplay(r.winPct)}</td>
                        <td className="px-3 py-2 text-right">{pctDisplay(r.makeCutPct)}</td>
                        <td className="px-3 py-2 text-right text-muted-foreground">{pctDisplay(r.top5Pct)}</td>
                        <td className="px-3 py-2 text-right text-muted-foreground">{pctDisplay(r.top10Pct)}</td>
                        <td className="px-3 py-2 text-right text-muted-foreground">{pctDisplay(r.top20Pct)}</td>
                        <td className={`px-3 py-2 text-right font-mono ${
                          (r.winDeltaPct ?? 0) > 0
                            ? 'text-emerald-400'
                            : (r.winDeltaPct ?? 0) < 0
                              ? 'text-red-400'
                              : 'text-muted-foreground'
                        }`}>
                          {signedPctDisplay(r.winDeltaPct)}
                        </td>
                        <td className="px-3 py-2 text-right text-muted-foreground">{pctDisplay(r.preWinPct)}</td>
                      </tr>
                      {open && (
                        <tr className="bg-secondary/20 border-b border-border/50">
                          <td colSpan={hasRoundUpdate ? 12 + (roundMeta?.rounds_completed ?? 0) : 10} className="px-4 py-3">
                            <div className="mb-4">
                              <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">
                                Current vs pre-tournament
                              </div>
                              <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
                                {[
                                  ['Win', pctDisplay(r.winPct), pctDisplay(r.preWinPct)],
                                  ['Make Cut', pctDisplay(r.makeCutPct), pctDisplay(r.preMakeCutPct)],
                                  ['Top 10', pctDisplay(r.top10Pct), pctDisplay(r.preTop10Pct)],
                                  ['Top 20', pctDisplay(r.top20Pct), pctDisplay(r.preTop20Pct)],
                                  ['SG/R', strokesDisplay(r.updatedSg), strokesDisplay(r.preSg)],
                                ].map(([label, current, pre]) => (
                                  <div key={label} className="rounded-md border border-border/50 bg-card/60 px-3 py-2">
                                    <div className="text-[10px] uppercase tracking-wider text-muted-foreground">{label}</div>
                                    <div className="text-sm font-mono font-semibold">{current}</div>
                                    <div className="text-[10px] text-muted-foreground mt-0.5">pre: {pre}</div>
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
              marketOdds={data.marketOdds ?? { tournament: eventName, fetchedAt: data.generatedAt, commenceTime: '', books: [], overrounds: {}, playerOdds: [] }}
              placementMarkets={data.placementMarkets}
            />
          ) : (
            <div className="rounded-lg border border-dashed border-border px-4 py-8 text-center text-muted-foreground text-sm">
              <p className="font-medium mb-2">No market odds data available</p>
              <p className="text-xs">
                Run{' '}
                <code className="bg-secondary px-1 rounded">python scripts/fetch_pga_odds.py</code>{' '}
                then{' '}
                <code className="bg-secondary px-1 rounded">python scripts/export_pga_tournament_dashboard.py</code>{' '}
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
