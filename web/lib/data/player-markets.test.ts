import { describe, expect, it } from "vitest";

import { deriveMlbHrBoardSnapshot } from "@/lib/data/player-markets";
import { deriveDataQuality } from "@/lib/data/data-quality";
import { summarizeMlbHr } from "@/lib/data/results";

type Run = Exclude<Parameters<typeof deriveMlbHrBoardSnapshot>[0], null>;
type BoardRow = Parameters<typeof deriveMlbHrBoardSnapshot>[1][number];

const baseRun: Run = {
  run_id: "run-1",
  run_key: "mlb-hr-2026-08-11-morning-1",
  slate_date: "2026-08-11",
  model_version: "mlb-hr-v1",
  run_window: "morning",
  status: "healthy",
  started_at: "2026-08-11T12:00:00Z",
  completed_at: "2026-08-11T14:00:00Z",
  workflow_url: null,
  gaps: [],
  validation_summary: null,
  total_candidates: 2,
  priced_candidates: 1,
  top25_denominator: 2,
  top25_priced_count: 1,
  top25_coverage: 0.5,
  prediction_ts: "2026-08-11T13:30:00Z",
  odds_ts: "2026-08-11T13:45:00Z",
};

function boardRow(overrides: Partial<BoardRow> = {}): BoardRow {
  return {
    board_row_id: "row-1",
    run_id: "run-1",
    run_key: baseRun.run_key,
    run_slate_date: baseRun.slate_date,
    run_window: baseRun.run_window,
    run_status: "healthy",
    run_completed_at: baseRun.completed_at,
    run_prediction_ts: baseRun.prediction_ts,
    run_odds_ts: baseRun.odds_ts,
    run_gaps: [],
    run_total_candidates: 2,
    run_priced_candidates: 1,
    run_top25_denominator: 2,
    run_top25_priced_count: 1,
    run_top25_coverage: 0.5,
    slate_date: baseRun.slate_date,
    game_id: "game-1",
    player_id: "player-1",
    player_name: "A. Batter",
    team: "AAA",
    opponent: "BBB",
    venue: "Example Park",
    event_time: "2026-08-11T23:00:00Z",
    lineup_slot: 2,
    lineup_status: "confirmed",
    opposing_probable_pitcher: "P. Pitcher",
    model_version: "mlb-hr-v1",
    model_probability: 0.31,
    baseline_probability: 0.2,
    rank: 1,
    book: "Book",
    american_price: 120,
    raw_market_probability: 0.45,
    no_vig_market_probability: 0.44,
    market_probability: 0.44,
    edge: -0.13,
    ev: -0.03,
    quarter_kelly: 0,
    odds_snapshot_ts: "2026-08-11T13:45:00Z",
    odds_status: "ok",
    odds_books_count: 3,
    quality_flags: [],
    statcast_available: true,
    statcast_coverage: 1,
    prediction_ts: baseRun.prediction_ts ?? "2026-08-11T13:30:00Z",
    published_at: baseRun.completed_at ?? "2026-08-11T14:00:00Z",
    ...overrides,
  };
}

describe("MLB HR trusted board snapshot", () => {
  it("fails closed before a current run exists", () => {
    const snapshot = deriveMlbHrBoardSnapshot(null, [], new Date("2026-08-11T12:00:00Z"));
    expect(snapshot.status).toBe("stale");
    expect(snapshot.rows).toHaveLength(0);
  });

  it("serves current rows and keeps model-only pricing blank", () => {
    const snapshot = deriveMlbHrBoardSnapshot(
      baseRun,
      [
        boardRow(),
        boardRow({ board_row_id: "row-2", player_id: "player-2", player_name: "B. Batter", rank: 2, book: null, american_price: null, market_probability: null, odds_status: "missing_odds" }),
      ],
      new Date("2026-08-11T16:00:00Z"),
    );
    expect(snapshot.status).toBe("healthy");
    expect(snapshot.counts.candidates).toBe(2);
    expect(snapshot.counts.priced).toBe(1);
    expect(snapshot.rows[1].edge).toBeNull();
  });

  it("shows partial coverage without hiding candidates", () => {
    const snapshot = deriveMlbHrBoardSnapshot(
      { ...baseRun, status: "partial" },
      [boardRow()],
      new Date("2026-08-11T16:00:00Z"),
    );
    expect(snapshot.status).toBe("partial");
    expect(snapshot.rows).toHaveLength(1);
  });

  it("requires an afternoon run after 4 PM", () => {
    const snapshot = deriveMlbHrBoardSnapshot(baseRun, [boardRow()], new Date("2026-08-11T23:00:00Z"));
    expect(snapshot.status).toBe("stale");
    expect(snapshot.rows).toHaveLength(0);
  });

  it("serves a healthy afternoon refresh immediately after the 2 PM threshold", () => {
    const afternoonRun: Run = {
      ...baseRun,
      run_key: "mlb-hr-2026-08-11-afternoon-101",
      run_window: "afternoon",
      started_at: "2026-08-11T20:30:00Z",
      completed_at: "2026-08-11T20:41:00Z",
      total_candidates: 120,
      priced_candidates: 113,
      top25_denominator: 25,
      top25_priced_count: 24,
      top25_coverage: 0.96,
    };
    const snapshot = deriveMlbHrBoardSnapshot(
      afternoonRun,
      [boardRow({ event_time: "2026-08-12T00:00:00Z" })],
      new Date("2026-08-11T20:45:00Z"),
    );

    expect(snapshot.status).toBe("healthy");
    expect(snapshot.counts.candidates).toBe(1);
    expect(snapshot.counts.priced).toBe(1);
  });

  it("represents a confirmed no-slate run explicitly", () => {
    const snapshot = deriveMlbHrBoardSnapshot({ ...baseRun, status: "no_slate" }, [], new Date("2026-08-11T16:00:00Z"));
    expect(snapshot.status).toBe("no_slate");
  });

  it("keeps zero coverage as unknown and accepts pass validation", () => {
    const rows = deriveDataQuality({
      generatedAt: "2026-08-11T12:00:00Z",
      oddspapi: { validation_status: "pass", validation_match_rate: 1 },
      records: [{
        sport: "MLB",
        modelVersion: "mlb-hr-v1",
        season: "2026",
        market: "home_run",
        sampleSize: 0,
        metrics: {},
        roi: null,
        units: null,
        bets: null,
        wins: null,
        losses: null,
        pushes: null,
        oddsStatus: "missing",
        sample: { odds_joined_games: 0, completed_games: 0 },
        artifactRefs: [],
        gaps: [],
        productionStatus: "candidate",
        productionGates: [],
      }],
      gaps: [],
    });
    expect(rows[0].status).toBe("ok");
    expect(rows[1].coveragePct).toBeNull();
  });

  it("separates priced ROI from model-only accuracy and voids", () => {
    const [summary] = summarizeMlbHr([
      { board_row_id: "1", game_date: "2026-08-10", player_name: "A", team: "AAA", model_version: "mlb-hr-v1", rank: 1, top_k_bucket: "top_10", model_probability: 0.3, actual_home_run: true, actual_plate_appearances: 4, american_price: 100, odds_status: "ok" },
      { board_row_id: null, game_date: "2026-08-10", player_name: "B", team: "AAA", model_version: "mlb-hr-v1", rank: 2, top_k_bucket: "top_10", model_probability: 0.2, actual_home_run: false, actual_plate_appearances: 4, american_price: null, odds_status: null },
      { board_row_id: null, game_date: "2026-08-10", player_name: "C", team: "AAA", model_version: "mlb-hr-v1", rank: 3, top_k_bucket: "top_10", model_probability: 0.1, actual_home_run: false, actual_plate_appearances: 0, american_price: 100, odds_status: "ok" },
    ]);
    expect(summary.pricedSample).toBe(1);
    expect(summary.roi).toBe(1);
    expect(summary.modelOnlySample).toBe(1);
    expect(summary.modelOnlyHitRate).toBe(0);
  });
});
