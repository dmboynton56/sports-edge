import { expect, it } from "vitest";

import type { GameResultRow } from "@/lib/data/results";
import {
  bookAtsResult,
  bookAtsSide,
  buildNflWeekReview,
  coverResult,
} from "@/lib/data/nfl-week-review";

const base: GameResultRow = {
  league: "NFL",
  season: 2026,
  week: 1,
  game_date: "2026-09-13",
  home_team: "KC",
  away_team: "DEN",
  home_score: 31,
  away_score: 10,
  book_spread: -3.5,
  my_spread: 2.5,
  my_home_win_prob: 0.44,
  model_version: "nfl-v2-live-20260906",
  spread_result: "loss",
  winner_result: "loss",
  flat_ats_units: -1,
};

it("grades home cover against a spread", () => {
  expect(coverResult(21, -3.5)).toBe("win");
  expect(coverResult(3, -3)).toBe("push");
  expect(coverResult(2, -3.5)).toBe("loss");
});

it("picks the sportsbook side from the model vs book number", () => {
  expect(bookAtsSide(-7, -3)).toBe("home");
  expect(bookAtsSide(-1, -3)).toBe("away");
  expect(bookAtsSide(-3, -3)).toBe("pass");
});

it("grades sportsbook ATS from the model's side of the number", () => {
  expect(bookAtsResult(21, 2.5, -3.5)).toBe("loss");
  expect(bookAtsResult(-13, 2.5, -3.5)).toBe("win");
  expect(bookAtsResult(7, -7, -3)).toBe("win");
  expect(bookAtsResult(3, -3, -3)).toBe("push");
});

it("builds a week recap with sportsbook ATS separate from model-spread cover", () => {
  const review = buildNflWeekReview([
    base,
    {
      ...base,
      game_date: "2026-09-13",
      home_team: "BUF",
      away_team: "BAL",
      home_score: 24,
      away_score: 20,
      book_spread: -2.5,
      my_spread: -4,
      my_home_win_prob: 0.62,
      spread_result: "win",
      winner_result: "win",
      flat_ats_units: 100 / 110,
    },
    { ...base, league: "NBA" },
    { ...base, week: 2 },
    { ...base, model_version: "v1" },
  ]);

  expect(review.games).toHaveLength(2);
  expect(review.winner).toMatchObject({ wins: 1, losses: 1, hitRate: 0.5 });
  expect(review.bookAts).toMatchObject({ wins: 1, losses: 1, hitRate: 0.5 });
  expect(review.games[0]?.bookAtsSide).toBe("home");
  expect(review.games[1]?.homeTeam).toBe("KC");
  expect(review.homeProbabilityBias).toBeCloseTo((0.44 + 0.62) / 2 - 1, 5);
});
