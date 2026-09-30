import { describe, expect, it } from "vitest";

import { formatDateTime } from "@/lib/format";

describe("game and forecast timestamps", () => {
  it("keeps Thursday night kickoff on Thursday in Mountain time", () => {
    expect(formatDateTime("2026-10-02T00:15:00Z")).toBe("Oct 1, 6:15 PM MDT");
  });

  it("keeps Monday night kickoff inside the Monday serving window", () => {
    expect(formatDateTime("2026-10-06T00:15:00Z")).toBe("Oct 5, 6:15 PM MDT");
  });

  it("uses the correct offset after daylight saving time ends", () => {
    expect(formatDateTime("2026-11-13T01:15:00Z")).toBe("Nov 12, 6:15 PM MST");
  });

  it("keeps unavailable and invalid timestamps explicit", () => {
    expect(formatDateTime(null)).toBe("n/a");
    expect(formatDateTime("invalid")).toBe("n/a");
  });
});
