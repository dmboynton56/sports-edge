export type BigQueryRuntimeConfig = {
  projectId?: string;
  credentialsConfigured: boolean;
};

export function getBigQueryRuntimeConfig(): BigQueryRuntimeConfig {
  return {
    projectId: process.env.BIGQUERY_PROJECT_ID ?? process.env.GCP_PROJECT_ID,
    credentialsConfigured: Boolean(process.env.GOOGLE_APPLICATION_CREDENTIALS),
  };
}

export function getBigQueryMissingEnv() {
  const config = getBigQueryRuntimeConfig();
  return [
    !config.projectId ? "BIGQUERY_PROJECT_ID or GCP_PROJECT_ID" : null,
    !config.credentialsConfigured ? "GOOGLE_APPLICATION_CREDENTIALS" : null,
  ].filter(Boolean) as string[];
}
