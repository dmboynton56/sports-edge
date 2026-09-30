import { afterEach, describe, expect, it, vi } from "vitest";
import { denverDate, nflPredictionIsCurrent, nflWeekWindow } from "@/lib/nfl-week";
import { deriveFreshness } from "@/lib/data/team-markets";

afterEach(() => vi.useRealTimers());

describe("weekly NFL forecast serving", () => {
  it("keeps Thursday, the weekend, and Monday in one cycle across month boundaries", () => {
    for (const day of ["2026-09-29", "2026-09-30", "2026-10-01", "2026-10-04", "2026-10-05"]) {
      expect(nflWeekWindow(day)).toEqual({ start: "2026-09-29", end: "2026-10-05" });
    }
    expect(nflWeekWindow("2026-10-06")).toEqual({ start: "2026-10-06", end: "2026-10-12" });
  });

  it("uses Denver midnight through the daylight saving transition", () => {
    expect(denverDate(new Date("2026-09-29T05:59:00Z"))).toBe("2026-09-28");
    expect(denverDate(new Date("2026-09-29T06:00:00Z"))).toBe("2026-09-29");
    expect(nflPredictionIsCurrent("2026-10-27T13:00:00Z", new Date("2026-11-02T15:00:00Z"))).toBe(true);
  });

  it("accepts Tuesday predictions on Monday, then rejects them in the next cycle", () => {
    const timestamp = "2026-09-29T13:05:00Z";
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-10-05T15:00:00Z"));
    expect(deriveFreshness(timestamp, -3, "NFL")).toBe("fresh");
    expect(deriveFreshness(timestamp, -3, "NBA")).toBe("stale");
    expect(deriveFreshness(timestamp, null, "NFL")).toBe("no_odds");
    vi.setSystemTime(new Date("2026-10-06T13:00:00Z"));
    expect(deriveFreshness(timestamp, -3, "NFL")).toBe("stale");
    expect(nflPredictionIsCurrent("invalid")).toBe(false);
    expect(nflPredictionIsCurrent("2026-10-06T14:00:00Z")).toBe(false);
  });
});
