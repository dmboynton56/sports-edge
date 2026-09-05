import { cache } from "react";

export type SupabaseRuntimeConfig = {
  url?: string;
  anonKey?: string;
  serviceRoleKeyConfigured: boolean;
  dbPasswordConfigured: boolean;
};

export function getSupabaseRuntimeConfig(): SupabaseRuntimeConfig {
  return {
    url: process.env.NEXT_PUBLIC_SUPABASE_URL ?? process.env.SUPABASE_URL,
    anonKey: process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY ?? process.env.SUPABASE_ANON_KEY,
    serviceRoleKeyConfigured: Boolean(process.env.SUPABASE_SERVICE_ROLE_KEY),
    dbPasswordConfigured: Boolean(
      process.env.SUPABASE_DB_PASSWORD ?? process.env.supabaseDBpass,
    ),
  };
}

export function getSupabaseMissingEnv() {
  const config = getSupabaseRuntimeConfig();
  const missing = [
    !config.url ? "NEXT_PUBLIC_SUPABASE_URL or SUPABASE_URL" : null,
    !config.anonKey ? "NEXT_PUBLIC_SUPABASE_ANON_KEY or SUPABASE_ANON_KEY" : null,
  ];
  return missing.filter((value): value is string => Boolean(value));
}

// Abort signals disable Next.js fetch memoization. React cache restores request-scoped
// deduplication within server renders; results are not retained between renders.
export const supabaseRest = cache(async function supabaseRest<T>(resource: string, revalidate = 300): Promise<T[] | null> {
  const config = getSupabaseRuntimeConfig();
  if (!config.url || !config.anonKey) return null;

  try {
    const base = config.url.replace(/\/$/, "");
    const response = await fetch(`${base}/rest/v1/${resource}`, {
      headers: {
        apikey: config.anonKey,
        Authorization: `Bearer ${config.anonKey}`,
      },
      next: { revalidate },
      signal: AbortSignal.timeout(8_000),
    });
    if (!response.ok) return null;
    const rows: unknown = await response.json();
    // SAFETY: Callers supply the selected row contract; reject non-array error payloads.
    return Array.isArray(rows) ? rows as T[] : null;
  } catch {
    return null;
  }
});
