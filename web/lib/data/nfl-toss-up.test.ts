import { expect, it } from "vitest";

import { isNflTossUp, nflMoneylinePick } from "./nfl-toss-up";

it("labels the ±0.05 band a toss-up and withholds a moneyline pick", () => {
  expect(isNflTossUp(0.494)).toBe(true);
  expect(isNflTossUp(0.45)).toBe(true);
  expect(isNflTossUp(0.55)).toBe(false);
  expect(nflMoneylinePick("DEN", "SEA", 0.494)).toBeNull();
  expect(nflMoneylinePick("DEN", "SEA", 0.364)).toBe("SEA");
  expect(nflMoneylinePick("DEN", "SEA", 0.6)).toBe("DEN");
});
