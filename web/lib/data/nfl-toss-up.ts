export const NFL_TOSS_UP_BAND = 0.05;

export function isNflTossUp(
  homeWinProb: number | null | undefined,
  band = NFL_TOSS_UP_BAND,
): boolean {
  return typeof homeWinProb === "number" && Number.isFinite(homeWinProb)
    && Math.abs(homeWinProb - 0.5) <= band;
}

export function nflMoneylinePick(
  homeTeam: string,
  awayTeam: string,
  homeWinProb: number | null | undefined,
  band = NFL_TOSS_UP_BAND,
): string | null {
  if (typeof homeWinProb !== "number" || !Number.isFinite(homeWinProb) || homeWinProb < 0 || homeWinProb > 1) {
    return null;
  }
  if (isNflTossUp(homeWinProb, band)) return null;
  if (homeWinProb === 0.5) return null;
  return homeWinProb > 0.5 ? homeTeam : awayTeam;
}
