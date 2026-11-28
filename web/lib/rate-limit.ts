export const RATE_LIMIT_WINDOW_MS = 60_000;
export const RATE_LIMIT_MAX = 60;

export type RateLimitBucket = {
  count: number;
  resetAt: number;
};

export type RateLimitDecision = {
  allowed: boolean;
  remaining: number;
  retryAfterSeconds: number;
  limit: number;
};

export const globalRateLimitStore = new Map<string, RateLimitBucket>();

export function clientIp(request: Request): string {
  const forwarded = request.headers.get("x-forwarded-for");
  if (forwarded) {
    const first = forwarded.split(",")[0]?.trim();
    if (first) return first;
  }
  return request.headers.get("x-real-ip")?.trim() || "anonymous";
}

function pruneStore(store: Map<string, RateLimitBucket>, now: number) {
  if (store.size < 8_000) return;
  for (const [key, bucket] of store) {
    if (now >= bucket.resetAt) store.delete(key);
  }
}

export function takeRateLimitToken(
  store: Map<string, RateLimitBucket>,
  key: string,
  now: number,
  limit: number = RATE_LIMIT_MAX,
  windowMs: number = RATE_LIMIT_WINDOW_MS,
): RateLimitDecision {
  pruneStore(store, now);
  const bucket = store.get(key);
  if (!bucket || now >= bucket.resetAt) {
    store.set(key, { count: 1, resetAt: now + windowMs });
    return {
      allowed: true,
      remaining: Math.max(0, limit - 1),
      retryAfterSeconds: Math.ceil(windowMs / 1000),
      limit,
    };
  }
  if (bucket.count >= limit) {
    return {
      allowed: false,
      remaining: 0,
      retryAfterSeconds: Math.max(1, Math.ceil((bucket.resetAt - now) / 1000)),
      limit,
    };
  }
  bucket.count += 1;
  return {
    allowed: true,
    remaining: Math.max(0, limit - bucket.count),
    retryAfterSeconds: Math.max(1, Math.ceil((bucket.resetAt - now) / 1000)),
    limit,
  };
}
