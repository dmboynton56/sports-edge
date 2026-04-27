'use client';

import { useState, useEffect } from 'react';
import { PickCard, type EnrichedPick } from '@/components/PickCard';
import {
  PlayersLeaderboard,
  type PlayerPrediction,
  type PlayerOdds,
  type MarketOddsDetail,
  type RecentTournamentRow,
} from '@/components/PlayersLeaderboard';
import { ModelMetrics, type ModelMetricsData } from '@/components/ModelMetrics';

// -----------------------------------------------------------------------
// NBA mock (existing)
// -----------------------------------------------------------------------
const MOCK_PICKS: EnrichedPick[] = [
  {
    id: "1",
    game: { id: "game1", league: "NBA", homeTeam: "Lakers", awayTeam: "Warriors", gameTimeUtc: "2026-03-12T02:30:00Z" },
    currentOdds: { book: "Consensus", market: "spread", line: -3.5, price: -110 },
    prediction: { predictedSpread: -5.5, homeWinProb: 0.68 },
    edgePts: 2.0
  },
  {
    id: "2",
    game: { id: "game2", league: "NBA", homeTeam: "Celtics", awayTeam: "Nuggets", gameTimeUtc: "2026-03-12T00:00:00Z" },
    currentOdds: { book: "Consensus", market: "spread", line: 2.5, price: -110 },
    prediction: { predictedSpread: 0.5, homeWinProb: 0.48 },
    edgePts: 2.0
  }
];

// -----------------------------------------------------------------------
// 2026 Players Championship — v2 model output (after R3)
// -----------------------------------------------------------------------
const PLAYERS_DATA: PlayerPrediction[] = [
  { name: "Ludvig Åberg", score: -14, expectedSG: -1.561, models: { ridge: -1.82, rf: -1.35, lgbm: -1.48, xgb: -1.60, nn: -1.56 }, mcWin: 0.534, mcTop5: 0.907, mcTop10: 0.981, clsTop10: 0.022, clsTop20: 0.062, clsWin: 0.001 },
  { name: "Michael Thorbjornsen", score: -10, expectedSG: -0.200, models: { ridge: -0.31, rf: -0.18, lgbm: -0.25, xgb: -0.15, nn: -0.11 }, mcWin: 0.045, mcTop5: 0.420, mcTop10: 0.710, clsTop10: 0.068, clsTop20: 0.130, clsWin: 0.004 },
  { name: "Cameron Young", score: -9, expectedSG: 0.091, models: { ridge: 0.12, rf: 0.05, lgbm: 0.08, xgb: 0.10, nn: 0.11 }, mcWin: 0.078, mcTop5: 0.494, mcTop10: 0.779, clsTop10: 0.123, clsTop20: 0.231, clsWin: 0.0084 },
  { name: "Brian Harman", score: -8, expectedSG: -0.176, models: { ridge: -0.20, rf: -0.22, lgbm: -0.15, xgb: -0.10, nn: -0.21 }, mcWin: 0.030, mcTop5: 0.297, mcTop10: 0.601, clsTop10: 0.102, clsTop20: 0.187, clsWin: 0.0077 },
  { name: "Matt Fitzpatrick", score: -8, expectedSG: 0.364, models: { ridge: 0.42, rf: 0.30, lgbm: 0.38, xgb: 0.35, nn: 0.37 }, mcWin: 0.046, mcTop5: 0.373, mcTop10: 0.681, clsTop10: 0.186, clsTop20: 0.287, clsWin: 0.022 },
  { name: "Viktor Hovland", score: -8, expectedSG: 0.762, models: { ridge: 0.85, rf: 0.68, lgbm: 0.72, xgb: 0.80, nn: 0.76 }, mcWin: 0.062, mcTop5: 0.441, mcTop10: 0.735, clsTop10: 0.286, clsTop20: 0.379, clsWin: 0.0295 },
  { name: "Corey Conners", score: -8, expectedSG: 0.164, models: { ridge: 0.18, rf: 0.12, lgbm: 0.15, xgb: 0.20, nn: 0.17 }, mcWin: 0.040, mcTop5: 0.344, mcTop10: 0.648, clsTop10: 0.142, clsTop20: 0.243, clsWin: 0.0137 },
  { name: "Justin Thomas", score: -8, expectedSG: 0.903, models: { ridge: 1.02, rf: 0.82, lgbm: 0.88, xgb: 0.95, nn: 0.85 }, mcWin: 0.071, mcTop5: 0.467, mcTop10: 0.754, clsTop10: 0.274, clsTop20: 0.406, clsWin: 0.0375 },
  { name: "Xander Schauffele", score: -8, expectedSG: 1.011, models: { ridge: 1.15, rf: 0.90, lgbm: 0.95, xgb: 1.05, nn: 1.01 }, mcWin: 0.076, mcTop5: 0.482, mcTop10: 0.769, clsTop10: 0.348, clsTop20: 0.451, clsWin: 0.0386 },
  { name: "Robert MacIntyre", score: -7, expectedSG: -1.121, models: { ridge: -1.25, rf: -1.05, lgbm: -1.10, xgb: -1.15, nn: -1.05 }, mcWin: 0.005, mcTop5: 0.096, mcTop10: 0.292, clsTop10: 0.031, clsTop20: 0.065, clsWin: 0.0027 },
  { name: "Sahith Theegala", score: -7, expectedSG: -0.819, models: { ridge: -0.90, rf: -0.75, lgbm: -0.82, xgb: -0.85, nn: -0.78 }, mcWin: 0.007, mcTop5: 0.115, mcTop10: 0.337, clsTop10: 0.049, clsTop20: 0.108, clsWin: 0.0026 },
  { name: "Austin Smotherman", score: -7, expectedSG: -0.429, models: { ridge: -0.50, rf: -0.38, lgbm: -0.45, xgb: -0.40, nn: -0.42 }, mcWin: 0.011, mcTop5: 0.151, mcTop10: 0.400, clsTop10: 0.078, clsTop20: 0.157, clsWin: 0.005 },
  { name: "Jacob Bridgeman", score: -7, expectedSG: -2.293, models: { ridge: -2.40, rf: -2.15, lgbm: -2.30, xgb: -2.35, nn: -2.26 }, mcWin: 0.002, mcTop5: 0.040, mcTop10: 0.159, clsTop10: 0.053, clsTop20: 0.116, clsWin: 0.0036 },
  { name: "Sepp Straka", score: -7, expectedSG: -0.691, models: { ridge: -0.75, rf: -0.62, lgbm: -0.70, xgb: -0.68, nn: -0.70 }, mcWin: 0.008, mcTop5: 0.126, mcTop10: 0.361, clsTop10: 0.052, clsTop20: 0.115, clsWin: 0.0018 },
  { name: "William Mouw", score: -6, expectedSG: -2.307, models: { ridge: -2.45, rf: -2.20, lgbm: -2.35, xgb: -2.28, nn: -2.26 }, mcWin: 0.000, mcTop5: 0.015, mcTop10: 0.081, clsTop10: 0.050, clsTop20: 0.113, clsWin: 0.0034 },
  { name: "Justin Rose", score: -6, expectedSG: 0.536, models: { ridge: 0.60, rf: 0.48, lgbm: 0.52, xgb: 0.55, nn: 0.53 }, mcWin: 0.010, mcTop5: 0.146, mcTop10: 0.391, clsTop10: 0.224, clsTop20: 0.362, clsWin: 0.0238 },
  { name: "Ryo Hisatsune", score: -6, expectedSG: -0.400, models: { ridge: -0.45, rf: -0.35, lgbm: -0.42, xgb: -0.38, nn: -0.40 }, mcWin: 0.004, mcTop5: 0.075, mcTop10: 0.250, clsTop10: 0.060, clsTop20: 0.130, clsWin: 0.003 },
  { name: "Russell Henley", score: -6, expectedSG: 0.083, models: { ridge: 0.10, rf: 0.05, lgbm: 0.08, xgb: 0.12, nn: 0.07 }, mcWin: 0.007, mcTop5: 0.111, mcTop10: 0.331, clsTop10: 0.140, clsTop20: 0.242, clsWin: 0.0107 },
  { name: "Keegan Bradley", score: -5, expectedSG: -0.199, models: { ridge: -0.22, rf: -0.18, lgbm: -0.20, xgb: -0.15, nn: -0.25 }, mcWin: 0.002, mcTop5: 0.043, mcTop10: 0.167, clsTop10: 0.094, clsTop20: 0.178, clsWin: 0.0064 },
  { name: "J.J. Spaun", score: -5, expectedSG: -1.185, models: { ridge: -1.30, rf: -1.10, lgbm: -1.18, xgb: -1.22, nn: -1.13 }, mcWin: 0.000, mcTop5: 0.018, mcTop10: 0.086, clsTop10: 0.040, clsTop20: 0.083, clsWin: 0.0015 },
  { name: "Brooks Koepka", score: -5, expectedSG: 0.138, models: { ridge: 0.15, rf: 0.10, lgbm: 0.12, xgb: 0.18, nn: 0.14 }, mcWin: 0.002, mcTop5: 0.055, mcTop10: 0.203, clsTop10: 0.139, clsTop20: 0.255, clsWin: 0.0128 },
  { name: "Alex Smalley", score: -5, expectedSG: -0.654, models: { ridge: -0.72, rf: -0.60, lgbm: -0.65, xgb: -0.68, nn: -0.62 }, mcWin: 0.001, mcTop5: 0.030, mcTop10: 0.127, clsTop10: 0.061, clsTop20: 0.134, clsWin: 0.0038 },
  { name: "Patrick Rodgers", score: -5, expectedSG: -0.598, models: { ridge: -0.65, rf: -0.55, lgbm: -0.60, xgb: -0.62, nn: -0.57 }, mcWin: 0.001, mcTop5: 0.029, mcTop10: 0.130, clsTop10: 0.061, clsTop20: 0.118, clsWin: 0.0037 },
  { name: "Maverick McNealy", score: -5, expectedSG: -0.231, models: { ridge: -0.25, rf: -0.20, lgbm: -0.23, xgb: -0.28, nn: -0.20 }, mcWin: 0.001, mcTop5: 0.041, mcTop10: 0.162, clsTop10: 0.128, clsTop20: 0.212, clsWin: 0.0075 },
  { name: "Scottie Scheffler", score: -4, expectedSG: 0.947, models: { ridge: 1.10, rf: 0.85, lgbm: 0.92, xgb: 0.98, nn: 0.89 }, mcWin: 0.002, mcTop5: 0.048, mcTop10: 0.183, clsTop10: 0.285, clsTop20: 0.430, clsWin: 0.0251 },
  { name: "Chris Gotterup", score: -4, expectedSG: 0.473, models: { ridge: 0.50, rf: 0.42, lgbm: 0.48, xgb: 0.45, nn: 0.52 }, mcWin: 0.001, mcTop5: 0.032, mcTop10: 0.136, clsTop10: 0.206, clsTop20: 0.309, clsWin: 0.0218 },
  { name: "Min Woo Lee", score: -4, expectedSG: -0.951, models: { ridge: -1.05, rf: -0.88, lgbm: -0.95, xgb: -0.98, nn: -0.90 }, mcWin: 0.000, mcTop5: 0.009, mcTop10: 0.049, clsTop10: 0.021, clsTop20: 0.061, clsWin: 0.0012 },
  { name: "Akshay Bhatia", score: -4, expectedSG: -0.927, models: { ridge: -1.00, rf: -0.85, lgbm: -0.92, xgb: -0.95, nn: -0.90 }, mcWin: 0.000, mcTop5: 0.008, mcTop10: 0.050, clsTop10: 0.037, clsTop20: 0.083, clsWin: 0.0014 },
  { name: "Keith Mitchell", score: -4, expectedSG: -0.167, models: { ridge: -0.18, rf: -0.12, lgbm: -0.15, xgb: -0.20, nn: -0.18 }, mcWin: 0.000, mcTop5: 0.018, mcTop10: 0.087, clsTop10: 0.098, clsTop20: 0.194, clsWin: 0.0073 },
  { name: "Jason Day", score: -4, expectedSG: 0.017, models: { ridge: 0.02, rf: -0.01, lgbm: 0.00, xgb: 0.05, nn: 0.02 }, mcWin: 0.001, mcTop5: 0.022, mcTop10: 0.100, clsTop10: 0.131, clsTop20: 0.238, clsWin: 0.0147 },
  { name: "Tommy Fleetwood", score: -4, expectedSG: -0.146, models: { ridge: -0.15, rf: -0.10, lgbm: -0.12, xgb: -0.18, nn: -0.18 }, mcWin: 0.000, mcTop5: 0.018, mcTop10: 0.092, clsTop10: 0.139, clsTop20: 0.237, clsWin: 0.012 },
  { name: "Si Woo Kim", score: -3, expectedSG: -0.179, models: { ridge: -0.20, rf: -0.15, lgbm: -0.18, xgb: -0.22, nn: -0.15 }, mcWin: 0.000, mcTop5: 0.006, mcTop10: 0.041, clsTop10: 0.097, clsTop20: 0.185, clsWin: 0.005 },
  { name: "Joe Highsmith", score: -3, expectedSG: -2.495, models: { ridge: -2.60, rf: -2.35, lgbm: -2.50, xgb: -2.55, nn: -2.48 }, mcWin: 0.000, mcTop5: 0.000, mcTop10: 0.005, clsTop10: 0.042, clsTop20: 0.102, clsWin: 0.0027 },
  { name: "Sam Burns", score: -3, expectedSG: -0.339, models: { ridge: -0.38, rf: -0.30, lgbm: -0.35, xgb: -0.32, nn: -0.30 }, mcWin: 0.000, mcTop5: 0.005, mcTop10: 0.035, clsTop10: 0.099, clsTop20: 0.165, clsWin: 0.0073 },
  { name: "Max Homa", score: -3, expectedSG: 0.045, models: { ridge: 0.05, rf: 0.02, lgbm: 0.04, xgb: 0.08, nn: 0.03 }, mcWin: 0.000, mcTop5: 0.008, mcTop10: 0.047, clsTop10: 0.105, clsTop20: 0.189, clsWin: 0.0061 },
];

// -----------------------------------------------------------------------
// Model evaluation metrics from v2 pipeline output
// -----------------------------------------------------------------------
const MODEL_METRICS: ModelMetricsData = {
  regression: [
    { model: 'Ridge', rmse: 1.8747, mae: 1.4379, spearman: 0.3927, ndcg10: 0.3429, ndcg20: 0.3960 },
    { model: 'Random Forest', rmse: 1.8534, mae: 1.4320, spearman: 0.3925, ndcg10: 0.3305, ndcg20: 0.3768 },
    { model: 'LightGBM', rmse: 1.8501, mae: 1.4312, spearman: 0.3882, ndcg10: 0.3399, ndcg20: 0.3792 },
    { model: 'XGBoost', rmse: 1.8571, mae: 1.4343, spearman: 0.3865, ndcg10: 0.2901, ndcg20: 0.3541 },
    { model: 'Neural Net (PyTorch)', rmse: 1.8547, mae: 1.4193, spearman: 0.4018, ndcg10: 0.3561, ndcg20: 0.3944 },
    { model: 'Meta Stack', rmse: 1.8348, mae: 1.4329, spearman: 0.4016, ndcg10: 0.3539, ndcg20: 0.3958 },
  ],
  metaWeights: [
    { model: 'Neural Net', weight: 0.759 },
    { model: 'LightGBM', weight: 0.282 },
    { model: 'Random Forest', weight: 0.205 },
    { model: 'Ridge', weight: -0.061 },
    { model: 'XGBoost', weight: -0.035 },
  ],
  classification: [
    {
      name: 'Made Cut',
      baseRate: 0.512,
      rows: [
        { model: 'Logistic Reg', brier: 0.2252, logLoss: 0.6415, auc: 0.6854, ece: 0.0389 },
        { model: 'Random Forest', brier: 0.2246, logLoss: 0.6400, auc: 0.6870, ece: 0.0342 },
        { model: 'LightGBM', brier: 0.2234, logLoss: 0.6368, auc: 0.6864, ece: 0.0309 },
        { model: 'XGBoost', brier: 0.2235, logLoss: 0.6371, auc: 0.6875, ece: 0.0355 },
        { model: 'Meta Stack', brier: 0.2219, logLoss: 0.6335, auc: 0.6904, ece: 0.0190 },
      ],
    },
    {
      name: 'Top 10',
      baseRate: 0.084,
      rows: [
        { model: 'Logistic Reg', brier: 0.0718, logLoss: 0.2613, auc: 0.7220, ece: 0.0069 },
        { model: 'Random Forest', brier: 0.0718, logLoss: 0.2614, auc: 0.7212, ece: 0.0062 },
        { model: 'LightGBM', brier: 0.0759, logLoss: 0.2832, auc: 0.7257, ece: 0.0265 },
        { model: 'XGBoost', brier: 0.0779, logLoss: 0.2847, auc: 0.7133, ece: 0.0417 },
        { model: 'Meta Stack', brier: 0.0725, logLoss: 0.2655, auc: 0.7319, ece: 0.0172 },
      ],
    },
    {
      name: 'Top 20',
      baseRate: 0.161,
      rows: [
        { model: 'Logistic Reg', brier: 0.1220, logLoss: 0.3970, auc: 0.7237, ece: 0.0201 },
        { model: 'Random Forest', brier: 0.1222, logLoss: 0.3973, auc: 0.7190, ece: 0.0167 },
        { model: 'LightGBM', brier: 0.1319, logLoss: 0.4321, auc: 0.7146, ece: 0.0535 },
        { model: 'XGBoost', brier: 0.1363, logLoss: 0.4392, auc: 0.6857, ece: 0.0731 },
        { model: 'Meta Stack', brier: 0.1224, logLoss: 0.3997, auc: 0.7230, ece: 0.0336 },
      ],
    },
    {
      name: 'Win',
      baseRate: 0.007,
      rows: [
        { model: 'Logistic Reg', brier: 0.0070, logLoss: 0.0365, auc: 0.8601, ece: 0.0008 },
        { model: 'Random Forest', brier: 0.0070, logLoss: 0.0369, auc: 0.8304, ece: 0.0009 },
        { model: 'LightGBM', brier: 0.0108, logLoss: 0.0630, auc: 0.7885, ece: 0.0336 },
        { model: 'XGBoost', brier: 0.0080, logLoss: 0.0467, auc: 0.7143, ece: 0.0039 },
        { model: 'Meta Stack', brier: 0.0071, logLoss: 0.0436, auc: 0.8522, ece: 0.0033 },
      ],
    },
  ],
  topFeatures: [
    { name: 'prev_sg_form_20', importance: 118999 },
    { name: 'prev_sg_form_10', importance: 60722 },
    { name: 'normal_wind_rounds_before', importance: 43950 },
    { name: 'relative_skill_vs_field', importance: 41034 },
    { name: 'field_strength_prev_avg_sg', importance: 37636 },
    { name: 'prev_avg_finish_num', importance: 34205 },
    { name: 'wind_premium_before', importance: 31468 },
    { name: 'prev_sg_form_3', importance: 27088 },
    { name: 'prev_sg_form_5', importance: 23102 },
    { name: 'prev_top20_rate', importance: 17075 },
  ],
};

// -----------------------------------------------------------------------
// Key takeaway cards
// -----------------------------------------------------------------------
const INSIGHTS = [
  {
    title: "McIlroy & Burns Co-Lead",
    body: "Both shot 67 (-5) in R1. McIlroy's model Expected SG is among the highest in the field — he's performing to his baseline. Burns is overperforming his model projection, suggesting regression risk in later rounds.",
    type: 'neutral' as const,
  },
  {
    title: "Scheffler Lurking",
    body: "At -2 (70) he's 3 back. But the model has Scheffler as the top Expected SG player in the field. This is a classic buy-low spot — strong model projection + reasonable deficit after R1.",
    type: 'buy' as const,
  },
  {
    title: "Reed & Day at -3",
    body: "Patrick Reed (69) and Jason Day (69) both outperformed expectations. Reed's model projection is moderate but his Augusta history is strong. Day's form has been inconsistent — the model suggests caution for Top 5.",
    type: 'buy' as const,
  },
  {
    title: "Rahm at +6: Fade",
    body: "Shot 78 in R1 — well below his model expectation. The model still rates his raw skill highly, but the 11-shot deficit makes any top-20 bet essentially dead money. Classic case where the math overrides reputation.",
    type: 'sell' as const,
  },
];

// -----------------------------------------------------------------------
// Page
// -----------------------------------------------------------------------
export default function Home() {
  const [mastersData, setMastersData] = useState<PlayerPrediction[]>([]);
  const [mastersRecentByPlayer, setMastersRecentByPlayer] = useState<
    Record<string, RecentTournamentRow[]>
  >({});
  const [loading, setLoading] = useState(true);
  const [mastersCalibNote, setMastersCalibNote] = useState<string | null>(null);
  const [tournamentStatus, setTournamentStatus] = useState<string>('PRE-TOURNAMENT');
  const [currentRound, setCurrentRound] = useState<number>(0);

  useEffect(() => {
    fetch('/data/pga_masters_dashboard.json')
      .then(r => r.json())
      .then(data => {
        const cal = data.predictionMeta?.calibration;
        if (cal?.note) setMastersCalibNote(cal.note as string);
        else setMastersCalibNote(null);

        const live = data.liveLeaderboard;
        const liveIndex: Record<string, { toPar: string; rounds: Record<string, number> }> = {};
        if (live?.players) {
          setTournamentStatus(live.status || 'In Progress');
          setCurrentRound(live.currentRound || 1);
          for (const lp of live.players) {
            liveIndex[lp.player] = { toPar: lp.toPar, rounds: lp.rounds };
          }
        }

        const predictions = data.predictions || [];
        const mapped: PlayerPrediction[] = predictions.map((p: any) => {
          let odds: PlayerOdds | undefined;
          if (p.market_implied_win != null && p.edge_win != null) {
            const otherMarkets: Record<string, MarketOddsDetail> = {};
            for (const mkt of ['top5', 'top10', 'top20', 'made_cut'] as const) {
              if (p[`market_implied_${mkt}`] != null && p[`edge_${mkt}`] != null) {
                otherMarkets[mkt] = {
                  marketImplied: p[`market_implied_${mkt}`],
                  bestPrice: p[`best_price_${mkt}`] ?? 0,
                  bestBook: p[`best_book_${mkt}`] ?? '',
                  edge: p[`edge_${mkt}`],
                  ev: p[`ev_${mkt}`] ?? 0,
                  kelly: p[`kelly_${mkt}`] ?? 0,
                  bookOdds: p[`book_odds_${mkt}`],
                };
              }
            }
            odds = {
              marketImplied: p.market_implied_win,
              bestPrice: p.best_price_win ?? 0,
              bestBook: p.best_book_win ?? '',
              edge: p.edge_win,
              ev: p.ev_win ?? 0,
              kelly: p.kelly_win ?? 0,
              bookOdds: p.book_odds_win,
              otherMarkets: Object.keys(otherMarkets).length > 0 ? otherMarkets : undefined,
            };
          }

          const livePlayer = liveIndex[p.player];
          let score = 0;
          if (livePlayer) {
            const tp = livePlayer.toPar.trim().toUpperCase();
            if (tp === 'E') score = 0;
            else {
              const n = parseInt(tp.replace('+', ''), 10);
              score = isNaN(n) ? 0 : n;
            }
          }

          return {
            name: p.player,
            score,
            expectedSG: p.exp_sg_per_round,
            models: {
              ridge: p.model_ridge,
              rf: p.model_rf,
              lgbm: p.model_lgbm,
              xgb: p.model_xgb,
              nn: p.model_nn
            },
            mcWin: p.sim_win_pct / 100,
            mcTop5: p.sim_top5_pct / 100,
            mcTop10: p.sim_top10_pct / 100,
            clsTop10: p.lr_target_top10_prob || 0,
            clsTop20: p.lr_target_top20_prob || 0,
            clsWin: p.lr_target_win_prob || 0,
            keyFeatures: p.key_features ? JSON.parse(p.key_features) : [],
            odds,
          };
        });
        setMastersData(mapped);
        setMastersRecentByPlayer(
          (data.recentByPlayer || {}) as Record<string, RecentTournamentRow[]>
        );
        setLoading(false);
      })
      .catch(err => {
        console.error(err);
        setLoading(false);
      });
  }, []);

  const topEdges = MOCK_PICKS.filter(p => p.edgePts > 0).sort((a, b) => b.edgePts - a.edgePts);
  const [showMetrics, setShowMetrics] = useState(false);

  return (
    <div className="container mx-auto px-4 py-8 max-w-7xl">
      <div className="flex flex-col space-y-10">

        {/* Header */}
        <div className="flex flex-col space-y-2">
          <h1 className="text-4xl font-bold tracking-tight">Sports Edge Dashboard</h1>
          <p className="text-muted-foreground text-lg">
            Live ML predictions — 8 models, 50K Monte Carlo simulations, RTX 5070 GPU accelerated.
          </p>
        </div>

        {/* Dashboard Stats */}
        <div className="grid gap-4 md:grid-cols-4">
          <div className="rounded-xl border border-border bg-card p-6 shadow-sm">
            <h3 className="text-sm font-medium text-muted-foreground">Active ML Models</h3>
            <div className="mt-2 text-3xl font-bold">8</div>
            <p className="text-xs text-muted-foreground mt-1">Ridge + RF + LGBM + XGB + NN + PyTorch Course Fit + Meta Stack + 4 Classifiers</p>
          </div>
          <div className="rounded-xl border border-border bg-card p-6 shadow-sm">
            <h3 className="text-sm font-medium text-muted-foreground">Monte Carlo Sims</h3>
            <div className="mt-2 text-3xl font-bold">50,000</div>
            <p className="text-xs text-muted-foreground mt-1">Per player, per round</p>
          </div>
          <div className="rounded-xl border border-border bg-card p-6 shadow-sm">
            <h3 className="text-sm font-medium text-muted-foreground">Ensemble Val RMSE</h3>
            <div className="mt-2 text-3xl font-bold text-accent">1.835</div>
            <p className="text-xs text-muted-foreground mt-1">Meta Stack (best of 6 models)</p>
          </div>
          <div className="rounded-xl border border-border bg-card p-6 shadow-sm">
            <h3 className="text-sm font-medium text-muted-foreground">Win Classifier AUC</h3>
            <div className="mt-2 text-3xl font-bold text-accent">0.860</div>
            <p className="text-xs text-muted-foreground mt-1">Logistic Regression (best calibrated)</p>
          </div>
        </div>

        {/* PGA Section */}
        <div>
          <div className="flex items-center justify-between mb-4 border-b border-border pb-3">
            <div className="flex items-center gap-3">
              <h2 className="text-2xl font-bold tracking-tight text-emerald-500">PGA Tour — The Masters Tournament</h2>
              <span className="bg-emerald-500/20 text-emerald-500 text-xs px-2.5 py-1 rounded-md font-bold">
                {currentRound > 0 ? `R${currentRound} — ${tournamentStatus}` : 'PRE-TOURNAMENT'}
              </span>
            </div>
            <span className="text-sm text-muted-foreground hidden md:block">Augusta National — Augusta, GA — April 9-12, 2026</span>
          </div>

          {/* Insight Cards */}
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4 mb-8">
            <div className="rounded-xl border border-emerald-500/30 bg-emerald-500/5 p-4">
              <h4 className="font-bold text-sm mb-1.5 text-emerald-400">Course Fit & Weather</h4>
              <p className="text-xs text-muted-foreground leading-relaxed">Dry conditions expected. The newly trained PyTorch Course Fit model heavily favors long hitters with strong recent baseline SG. Scheffler and Rahm strongly benefit.</p>
            </div>
            <div className="rounded-xl border border-emerald-500/30 bg-emerald-500/5 p-4">
              <h4 className="font-bold text-sm mb-1.5 text-emerald-400">Scheffler Dominance</h4>
              <p className="text-xs text-muted-foreground leading-relaxed">The model confirms Scottie Scheffler's extreme edge. His expected SG is dominant across all models.</p>
            </div>
            <div className="rounded-xl border border-red-500/30 bg-red-500/5 p-4">
              <h4 className="font-bold text-sm mb-1.5 text-red-400">Tiger Woods</h4>
              <p className="text-xs text-muted-foreground leading-relaxed">Lack of recent volume and competitive form significantly drags down Expected SG. The model gives a very low make cut probability.</p>
            </div>
            <div className="rounded-xl border border-border bg-card p-4">
              <h4 className="font-bold text-sm mb-1.5 text-foreground">Augusta Intangibles</h4>
              <p className="text-xs text-muted-foreground leading-relaxed">Experience at Augusta is factored into the PyTorch embeddings, elevating past champions and those with strong course history.</p>
            </div>
          </div>

          {/* Full Leaderboard */}
          <div className="mb-8">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-lg font-semibold">
                {currentRound > 0 ? `Leaderboard After R${currentRound}` : 'Full Leaderboard'} — Model Predictions
              </h3>
              <span className="text-xs text-muted-foreground">Sorted by To Par</span>
            </div>
            <p className="text-xs text-muted-foreground mb-3 max-w-4xl leading-relaxed">
              Primary read: Monte Carlo win / top-10% (field-relative). Logistic win% is trained on in-distribution rows and can spike for players with unusual tour mixes (e.g. LIV-heavy); treat it as a secondary signal.
              {mastersCalibNote ? ` ${mastersCalibNote}` : ''}
            </p>
            {loading ? (
              <div className="text-sm text-muted-foreground py-10 text-center border rounded-xl border-dashed">Loading ML predictions...</div>
            ) : (
              <PlayersLeaderboard players={mastersData} recentByPlayer={mastersRecentByPlayer} />
            )}
          </div>

          {/* Model Metrics Toggle */}
          <div className="mb-8">
            <button
              onClick={() => setShowMetrics(!showMetrics)}
              className="flex items-center gap-2 text-sm font-medium text-muted-foreground hover:text-foreground transition-colors mb-4"
            >
              <svg
                className={`w-4 h-4 transition-transform duration-200 ${showMetrics ? 'rotate-180' : ''}`}
                fill="none" viewBox="0 0 24 24" stroke="currentColor"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
              {showMetrics ? 'Hide' : 'Show'} Model Evaluation & Metrics
            </button>
            {showMetrics && <ModelMetrics data={MODEL_METRICS} />}
          </div>
        </div>

        {/* NBA/NFL Section */}
        <div className="mt-4">
          <div className="flex items-center justify-between mb-4 border-b border-border pb-2">
            <h2 className="text-2xl font-bold tracking-tight">NBA / NFL Spread Edges</h2>
          </div>
          <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
            {topEdges.map(pick => (
              <PickCard key={pick.id} pick={pick} />
            ))}
          </div>
        </div>

      </div>
    </div>
  );
}
