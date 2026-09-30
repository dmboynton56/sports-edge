/** Convert a model probability to a no-vig American fair price for display. */
export function formatFairAmericanOdds(probability: number | null | undefined): string {
  if (probability == null || !Number.isFinite(probability) || probability <= 0 || probability >= 1) {
    return "—";
  }
  const american = probability <= 0.5
    ? 100 * (1 - probability) / probability
    : -100 * probability / (1 - probability);
  if (!Number.isFinite(american)) return "—";
  return american > 0 ? `+${Math.round(american)}` : `${Math.round(american)}`;
}

export function isPlayerProbabilityMarket(market: string): boolean {
  return market === "home_run" || market === "anytime_td";
}
