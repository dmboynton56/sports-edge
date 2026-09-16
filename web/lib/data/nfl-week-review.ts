import type { GameResultRow } from "@/lib/data/results";

export const NFL_LIVE_MODEL_VERSION = "nfl-v2-live-20260906";
export const NFL_WEEK1_SEASON = 2026;
export const NFL_WEEK1_WEEK = 1;

export type CoverResult = "win" | "loss" | "push";
export type BookAtsSide = "home" | "away" | "pass";

export type NflWeekGame = {
  gameDate: string;
  homeTeam: string;
  awayTeam: string;
  homeScore: number;
  awayScore: number;
  actualMargin: number;
  mySpread: number | null;
  bookSpread: number | null;
  homeWinProb: number | null;
  winnerPick: "home" | "away" | null;
  winnerResult: CoverResult | null;
  modelSpreadResult: CoverResult | null;
  bookAtsSide: BookAtsSide | null;
  bookAtsResult: CoverResult | null;
  marginError: number | null;
};

export type NflWeekRecord = {
  sample: number;
  wins: number;
  losses: number;
  pushes: number;
  hitRate: number | null;
  units: number | null;
};

export type NflWeekReview = {
  season: number;
  week: number;
  modelVersion: string;
  games: NflWeekGame[];
  winner: NflWeekRecord;
  modelSpread: NflWeekRecord;
  bookAts: NflWeekRecord;
  avgHomeWinProb: number | null;
  actualHomeWinRate: number | null;
  homeProbabilityBias: number | null;
  spreadMae: number | null;
};

function rate(wins: number, losses: number) {
  const risked = wins + losses;
  return risked ? wins / risked : null;
}

function record(results: Array<CoverResult | null>): NflWeekRecord {
  const wins = results.filter((result) => result === "win").length;
  const losses = results.filter((result) => result === "loss").length;
  const pushes = results.filter((result) => result === "push").length;
  const units = results.reduce((sum, result) => {
    if (result === "win") return sum + 100 / 110;
    if (result === "loss") return sum - 1;
    return sum;
  }, 0);
  const sample = wins + losses + pushes;
  return {
    sample,
    wins,
    losses,
    pushes,
    hitRate: rate(wins, losses),
    units: sample ? units : null,
  };
}

export function coverResult(actualMargin: number, spread: number): CoverResult {
  const cover = actualMargin + spread;
  if (Math.abs(cover) < 0.001) return "push";
  return cover > 0 ? "win" : "loss";
}

export function bookAtsSide(mySpread: number, bookSpread: number): BookAtsSide {
  if (Math.abs(mySpread - bookSpread) < 0.001) return "pass";
  return mySpread < bookSpread ? "home" : "away";
}

export function bookAtsResult(
  actualMargin: number,
  mySpread: number,
  bookSpread: number,
): CoverResult {
  const side = bookAtsSide(mySpread, bookSpread);
  const homeCovers = coverResult(actualMargin, bookSpread);
  if (side === "pass" || homeCovers === "push") return "push";
  if (side === "home") return homeCovers;
  return homeCovers === "win" ? "loss" : "win";
}

export function winnerPick(homeWinProb: number | null): "home" | "away" | null {
  if (homeWinProb == null || !Number.isFinite(homeWinProb)) return null;
  return homeWinProb >= 0.5 ? "home" : "away";
}

export function filterNflWeek(
  rows: GameResultRow[],
  season = NFL_WEEK1_SEASON,
  week = NFL_WEEK1_WEEK,
  modelVersion = NFL_LIVE_MODEL_VERSION,
): GameResultRow[] {
  return rows.filter(
    (row) =>
      row.league === "NFL"
      && Number(row.season) === season
      && Number(row.week) === week
      && row.model_version === modelVersion,
  );
}

export function buildNflWeekReview(
  rows: GameResultRow[],
  season = NFL_WEEK1_SEASON,
  week = NFL_WEEK1_WEEK,
  modelVersion = NFL_LIVE_MODEL_VERSION,
): NflWeekReview {
  const games = filterNflWeek(rows, season, week, modelVersion)
    .map((row): NflWeekGame => {
      const actualMargin = row.home_score - row.away_score;
      const pick = winnerPick(row.my_home_win_prob);
      const atsSide =
        row.my_spread != null && row.book_spread != null
          ? bookAtsSide(row.my_spread, row.book_spread)
          : null;
      return {
        gameDate: row.game_date,
        homeTeam: row.home_team,
        awayTeam: row.away_team,
        homeScore: row.home_score,
        awayScore: row.away_score,
        actualMargin,
        mySpread: row.my_spread,
        bookSpread: row.book_spread,
        homeWinProb: row.my_home_win_prob,
        winnerPick: pick,
        winnerResult: row.winner_result,
        modelSpreadResult: row.spread_result,
        bookAtsSide: atsSide,
        bookAtsResult:
          row.my_spread == null || row.book_spread == null
            ? null
            : bookAtsResult(actualMargin, row.my_spread, row.book_spread),
        marginError: row.my_spread == null ? null : Math.abs(actualMargin + row.my_spread),
      };
    })
    .toSorted((left, right) => {
      const date = left.gameDate.localeCompare(right.gameDate);
      if (date) return date;
      return left.homeTeam.localeCompare(right.homeTeam);
    });

  const homeWins = games.filter((game) => game.homeScore > game.awayScore).length;
  const probs = games
    .map((game) => game.homeWinProb)
    .filter((value): value is number => value != null && Number.isFinite(value));
  const avgHomeWinProb = probs.length
    ? probs.reduce((sum, value) => sum + value, 0) / probs.length
    : null;
  const actualHomeWinRate = games.length ? homeWins / games.length : null;
  const spreadErrors = games
    .map((game) => game.marginError)
    .filter((value): value is number => value != null);

  return {
    season,
    week,
    modelVersion,
    games,
    winner: record(games.map((game) => game.winnerResult)),
    modelSpread: record(games.map((game) => game.modelSpreadResult)),
    bookAts: record(games.map((game) => game.bookAtsResult)),
    avgHomeWinProb,
    actualHomeWinRate,
    homeProbabilityBias:
      avgHomeWinProb != null && actualHomeWinRate != null
        ? avgHomeWinProb - actualHomeWinRate
        : null,
    spreadMae: spreadErrors.length
      ? spreadErrors.reduce((sum, value) => sum + value, 0) / spreadErrors.length
      : null,
  };
}
