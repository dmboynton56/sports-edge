import type { JsonValue } from "@/lib/data/json";

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

export function asRestRows<T>(payload: JsonValue): T[] | null {
  // SAFETY: Callers supply the row contract for a matching Supabase select and this boundary verifies the response is an array.
  return Array.isArray(payload) ? (payload as T[]) : null;
}

export async function supabaseRest<T>(resource: string): Promise<T[] | null> {
  const config = getSupabaseRuntimeConfig();
  if (!config.url || !config.anonKey) return null;

  try {
    const base = config.url.replace(/\/$/, "");
    const response = await fetch(`${base}/rest/v1/${resource}`, {
      headers: {
        apikey: config.anonKey,
        Authorization: `Bearer ${config.anonKey}`,
      },
      next: { revalidate: 300 },
    });
    if (!response.ok) return null;
    const payload = await response.json();
    return asRestRows<T>(payload);
  } catch {
    return null;
  }
}
