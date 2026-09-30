import { describe, expect, it } from "vitest";

import { formatFairAmericanOdds } from "@/lib/fair-odds";

describe("model fair odds", () => {
  it("converts valid probabilities without a sportsbook margin", () => {
    expect(formatFairAmericanOdds(0.2)).toBe("+400");
    expect(formatFairAmericanOdds(0.5)).toBe("+100");
    expect(formatFairAmericanOdds(0.8)).toBe("-400");
  });

  it("rejects missing or invalid probabilities", () => {
    expect(formatFairAmericanOdds(null)).toBe("—");
    expect(formatFairAmericanOdds(0)).toBe("—");
    expect(formatFairAmericanOdds(1)).toBe("—");
  });
});
