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
  return [
    !config.url ? "NEXT_PUBLIC_SUPABASE_URL or SUPABASE_URL" : null,
    !config.anonKey ? "NEXT_PUBLIC_SUPABASE_ANON_KEY or SUPABASE_ANON_KEY" : null,
  ].filter(Boolean) as string[];
}
