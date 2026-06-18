# Sports Edge Scheduler Trigger

This Cloud Run service receives authenticated Cloud Scheduler HTTP requests and
triggers GitHub Actions workflows through `workflow_dispatch`.

## Deploy

Set these once:

```bash
export PROJECT_ID="learned-pier-478122-p7"
export REGION="us-central1"
export REPO_OWNER="dmboynton56"
export REPO_NAME="sports-edge"
export GITHUB_REF="main"
```

Create a fine-grained GitHub token with access to this repository and
`Actions: Read and write`, then store it in Secret Manager:

```bash
gcloud services enable run.googleapis.com cloudbuild.googleapis.com secretmanager.googleapis.com cloudscheduler.googleapis.com

printf "%s" "$GITHUB_ACTIONS_DISPATCH_TOKEN" | gcloud secrets create sports-edge-github-actions-token \
  --project "$PROJECT_ID" \
  --data-file=-
```

Deploy the trigger:

```bash
gcloud run deploy sports-edge-scheduler-trigger \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --source gcp/scheduler-trigger \
  --no-allow-unauthenticated \
  --set-env-vars "GITHUB_OWNER=$REPO_OWNER,GITHUB_REPO=$REPO_NAME,GITHUB_REF=$GITHUB_REF" \
  --set-secrets "GITHUB_TOKEN=sports-edge-github-actions-token:latest"
```

Grant the Cloud Run runtime service account access to the secret if your deploy
command reports that it is missing permission:

```bash
PROJECT_NUMBER="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"
RUNTIME_SA="$PROJECT_NUMBER-compute@developer.gserviceaccount.com"

gcloud secrets add-iam-policy-binding sports-edge-github-actions-token \
  --project "$PROJECT_ID" \
  --member "serviceAccount:$RUNTIME_SA" \
  --role "roles/secretmanager.secretAccessor"
```

## Scheduler Jobs

Create a dedicated Scheduler caller and allow it to invoke the Cloud Run service:

```bash
gcloud iam service-accounts create sports-edge-scheduler \
  --project "$PROJECT_ID" \
  --display-name "Sports Edge Cloud Scheduler"

SCHEDULER_SA="sports-edge-scheduler@$PROJECT_ID.iam.gserviceaccount.com"

gcloud run services add-iam-policy-binding sports-edge-scheduler-trigger \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --member "serviceAccount:$SCHEDULER_SA" \
  --role "roles/run.invoker"

TRIGGER_URL="$(gcloud run services describe sports-edge-scheduler-trigger \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --format='value(status.url)')"
```

Create the jobs:

```bash
gcloud scheduler jobs create http sports-edge-daily-refresh \
  --project "$PROJECT_ID" \
  --location "$REGION" \
  --schedule "0 13 * * *" \
  --time-zone "Etc/UTC" \
  --uri "$TRIGGER_URL/dispatch/daily-refresh" \
  --http-method POST \
  --headers "Content-Type=application/json" \
  --message-body '{}' \
  --oidc-service-account-email "$SCHEDULER_SA" \
  --oidc-token-audience "$TRIGGER_URL"

gcloud scheduler jobs create http sports-edge-player-markets-refresh \
  --project "$PROJECT_ID" \
  --location "$REGION" \
  --schedule "30 12 * * *" \
  --time-zone "Etc/UTC" \
  --uri "$TRIGGER_URL/dispatch/player-markets-refresh" \
  --http-method POST \
  --headers "Content-Type=application/json" \
  --message-body '{}' \
  --oidc-service-account-email "$SCHEDULER_SA" \
  --oidc-token-audience "$TRIGGER_URL"

gcloud scheduler jobs create http sports-edge-world-cup-refresh \
  --project "$PROJECT_ID" \
  --location "$REGION" \
  --schedule "0 3,13,17,21 * * *" \
  --time-zone "Etc/UTC" \
  --uri "$TRIGGER_URL/dispatch/world-cup-refresh" \
  --http-method POST \
  --headers "Content-Type=application/json" \
  --message-body '{}' \
  --oidc-service-account-email "$SCHEDULER_SA" \
  --oidc-token-audience "$TRIGGER_URL"
```

Run one manually:

```bash
gcloud scheduler jobs run sports-edge-daily-refresh \
  --project "$PROJECT_ID" \
  --location "$REGION"
```
