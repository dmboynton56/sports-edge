import { describe, expect, it } from "vitest";

import { formatCfbMarketSubject } from "@/lib/data/cfb-markets";

describe("CFB market labels", () => {
  it("attaches the matchup to totals so over and under rows are identifiable", () => {
    expect(formatCfbMarketSubject("Over 50.5", "total", "Virginia Tech", "South Carolina"))
      .toBe("Over 50.5 · Virginia Tech @ South Carolina");
    expect(formatCfbMarketSubject("Under 58.5", "total", "TCU", "North Carolina"))
      .toBe("Under 58.5 · TCU @ North Carolina");
  });

  it("keeps team-market subjects unchanged", () => {
    expect(formatCfbMarketSubject("South Alabama Jaguars -13.5", "spread", "Morgan State", "South Alabama"))
      .toBe("South Alabama Jaguars -13.5");
  });
});
