'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';

type PredRow = {
  player: string;
  exp_sg_per_round: number;
  sim_win_pct: number;
  sim_top5_pct: number;
  sim_top10_pct: number;
  sim_top20_pct: number;
  lr_target_made_cut_prob?: number;
  lr_target_top10_prob?: number;
  lr_target_top20_prob?: number;
  lr_target_win_prob?: number;
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

type Dashboard = {
  generatedAt: string;
  predictions: PredRow[];
  predictionMeta: Record<string, unknown>;
  espnSupplement: { path: string; rows: number; seasons?: number[] };
  tournaments2026: TournRow[];
  recentByPlayer: Record<string, FormRow[]>;
};

type Tab = 'predictions' | 'schedule' | 'form';

const DATA_URL = '/data/pga_masters_dashboard.json';

function pctProb(x: number | undefined) {
  if (x === undefined || Number.isNaN(x)) return '—';
  return `${(x * 100).toFixed(1)}%`;
}

export default function PGAPage() {
  const [data, setData] = useState<Dashboard | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>('predictions');
  const [sortKey, setSortKey] = useState<keyof PredRow>('sim_win_pct');
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc');
  const [playerPick, setPlayerPick] = useState<string>('');

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
    (k: keyof PredRow) => {
      if (sortKey === k) setSortDir((d) => (d === 'desc' ? 'asc' : 'desc'));
      else {
        setSortKey(k);
        setSortDir('desc');
      }
    },
    [sortKey]
  );

  const formRows = data?.recentByPlayer[playerPick] ?? [];

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
            ['predictions', 'Predictions'],
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

      {tab === 'predictions' && data && (
        <div className="rounded-xl border border-border bg-card overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border bg-secondary/30">
                  <th className="text-left px-3 py-2 font-medium text-muted-foreground">#</th>
                  <th className="text-left px-3 py-2 font-medium text-muted-foreground">Player</th>
                  {(
                    [
                      ['exp_sg_per_round', 'Exp SG/R'],
                      ['sim_win_pct', 'MC Win%'],
                      ['sim_top5_pct', 'Top5%'],
                      ['sim_top10_pct', 'Top10%'],
                      ['sim_top20_pct', 'Top20%'],
                      ['lr_target_made_cut_prob', 'P(cut)'],
                      ['lr_target_top10_prob', 'LR T10'],
                      ['lr_target_top20_prob', 'LR T20'],
                      ['lr_target_win_prob', 'LR Win'],
                    ] as const
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
                {sortedPreds.map((r, i) => (
                  <tr key={r.player} className="border-b border-border/50 hover:bg-secondary/10">
                    <td className="px-3 py-2 text-muted-foreground">{i + 1}</td>
                    <td className="px-3 py-2 font-medium">{r.player}</td>
                    <td className="px-3 py-2 text-right font-mono text-emerald-400/90">
                      {r.exp_sg_per_round >= 0 ? '+' : ''}
                      {r.exp_sg_per_round.toFixed(3)}
                    </td>
                    <td className="px-3 py-2 text-right">{r.sim_win_pct.toFixed(1)}%</td>
                    <td className="px-3 py-2 text-right text-muted-foreground">{r.sim_top5_pct.toFixed(1)}%</td>
                    <td className="px-3 py-2 text-right text-muted-foreground">{r.sim_top10_pct.toFixed(1)}%</td>
                    <td className="px-3 py-2 text-right text-muted-foreground">{r.sim_top20_pct.toFixed(1)}%</td>
                    <td className="px-3 py-2 text-right">{pctProb(r.lr_target_made_cut_prob)}</td>
                    <td className="px-3 py-2 text-right">{pctProb(r.lr_target_top10_prob)}</td>
                    <td className="px-3 py-2 text-right">{pctProb(r.lr_target_top20_prob)}</td>
                    <td className="px-3 py-2 text-right">{pctProb(r.lr_target_win_prob)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
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
