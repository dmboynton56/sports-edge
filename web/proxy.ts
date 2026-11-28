import { NextResponse, type NextRequest } from "next/server";

import {
  RATE_LIMIT_MAX,
  RATE_LIMIT_WINDOW_MS,
  clientIp,
  globalRateLimitStore,
  takeRateLimitToken,
} from "@/lib/rate-limit";

export function proxy(request: NextRequest) {
  const decision = takeRateLimitToken(
    globalRateLimitStore,
    `public-api:${clientIp(request)}`,
    Date.now(),
    RATE_LIMIT_MAX,
    RATE_LIMIT_WINDOW_MS,
  );
  if (!decision.allowed) {
    return NextResponse.json(
      { error: "rate_limited", retryAfterSeconds: decision.retryAfterSeconds },
      {
        status: 429,
        headers: {
          "Retry-After": String(decision.retryAfterSeconds),
          "X-RateLimit-Limit": String(decision.limit),
          "X-RateLimit-Remaining": "0",
        },
      },
    );
  }
  const response = NextResponse.next();
  response.headers.set("X-RateLimit-Limit", String(decision.limit));
  response.headers.set("X-RateLimit-Remaining", String(decision.remaining));
  return response;
}

export const config = {
  matcher: ["/api/mcp", "/api/mcp/:path*", "/api/mobile/:path*"],
};
