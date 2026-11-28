import { describe, expect, it } from "vitest";

import { clientIp, takeRateLimitToken } from "@/lib/rate-limit";

describe("clientIp", () => {
  it("prefers the first x-forwarded-for hop", () => {
    const request = new Request("https://example.test/api/mcp", {
      headers: {
        "x-forwarded-for": "203.0.113.9, 10.0.0.1",
        "x-real-ip": "10.0.0.1",
      },
    });
    expect(clientIp(request)).toBe("203.0.113.9");
  });
});

describe("takeRateLimitToken", () => {
  it("resets after the window elapses", () => {
    const store = new Map();
    expect(takeRateLimitToken(store, "a", 0, 1, 1_000).allowed).toBe(true);
    expect(takeRateLimitToken(store, "a", 10, 1, 1_000).allowed).toBe(false);
    expect(takeRateLimitToken(store, "a", 1_000, 1, 1_000).allowed).toBe(true);
  });
});
