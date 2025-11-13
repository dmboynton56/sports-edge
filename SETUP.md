# Sports-Edge: Setup Instructions

## Prerequisites

1. **API Keys**:
   - The Odds API key: https://the-odds-api.com/
   - Supabase project URL and service role key

2. **Python Environment**:
   - Python 3.9+
   - Virtual environment recommended

## Setup Steps

### 1. Install Dependencies

```bash
cd sports-edge
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your API keys
```

Optional overrides:

- `SUPABASE_DB_HOST` – explicitly set your Supabase Postgres host (useful if your project uses a custom domain or the default `db.<ref>.supabase.co` host does not resolve).
- `SUPABASE_DB_PORT` – custom Postgres port (defaults to `5432`).
- `SUPABASE_DB_NAME` – database name (defaults to `postgres`).
- `SUPABASE_DB_USER` – database user/tenant (defaults to `postgres`; some pooler strings require `postgres.<project-ref>`).

### 3. Set Up Supabase Database

1. Go to your Supabase project SQL editor
2. Run the migration scripts in order (e.g., `sql/001_initial_schema.sql`, `sql/002_add_week_column.sql`, `sql/003_add_book_spread.sql`)
3. Verify tables and views are created

### 4. Train Initial Models (Optional)

Models can be trained via notebooks or separate training scripts. For MVP, you can use simple baseline models that will be created on first run.

### 5. Test the Pipeline

```bash
# Test NFL refresh
python -m src.pipeline.refresh --league NFL --date 2025-11-06

# Test NBA refresh
python -m src.pipeline.refresh --league NBA --date 2025-11-06
```

### 6. Set Up GitHub Actions (Optional)

1. Add secrets to your GitHub repository:
   - `ODDS_API_KEY`
   - `SUPABASE_URL`
   - `SUPABASE_SERVICE_ROLE_KEY`

2. The workflow will run automatically every 15 minutes during game days

### 7. Verify Next.js Integration

1. Ensure Supabase environment variables are set in `personal-portfolio/.env.local`:
   ```
   SUPABASE_URL=your_supabase_url
   SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
   ```

2. Start the Next.js dev server:
   ```bash
   cd personal-portfolio
   npm run dev
   ```

3. Navigate to the Work section and verify Sports Edge card appears

## Troubleshooting

- **No games found**: Check that the date is during the season and API keys are valid
- **Database errors**: Verify Supabase schema is migrated and RLS policies are set
- **Model errors**: Ensure model files exist in `models/` directory or models will be skipped

