import argparse
import os
import sys
from datetime import date
from typing import List, Optional, Tuple
from urllib.parse import urlparse

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import psycopg
from dotenv import load_dotenv

from src.data import nfl_fetcher
from src.data.pbp_loader import load_pbp
from src.models.predictor import GamePredictor


MODEL_VERSION = 'v2'


def load_season_schedule(season: int) -> pd.DataFrame:
    """Fetch the NFL schedule for a season and normalize columns."""
    print(f"Loading {season} NFL season data...")
    schedule = nfl_fetcher.fetch_nfl_schedule(season)
    if 'gameday' in schedule.columns:
        schedule['game_date'] = pd.to_datetime(schedule['gameday'])
    elif 'game_date' in schedule.columns:
        schedule['game_date'] = pd.to_datetime(schedule['game_date'])
    else:
        raise ValueError("Schedule missing game_date/gameday column.")
    schedule['season'] = season
    print(f"  Loaded {len(schedule)} games for {season}")
    return schedule


def filter_completed_games(schedule: pd.DataFrame) -> pd.DataFrame:
    """Return only games that have final scores logged."""
    if 'home_score' not in schedule.columns or 'away_score' not in schedule.columns:
        raise ValueError("Schedule is missing score columns needed to identify completed games.")
    
    completed = schedule[
        schedule['home_score'].notna() & schedule['away_score'].notna()
    ].copy()
    
    print(f"  Completed games: {len(completed)}")
    if len(completed) > 0:
        print(f"  Completed through: {completed['game_date'].max().date()}")
    else:
        print("  WARNING: No completed games yet; team-strength features will be empty.")
    
    return completed


def collect_week_games(schedule: pd.DataFrame, week: int) -> pd.DataFrame:
    """
    Gather games for the specified week.
    """
    if 'week' not in schedule.columns:
        raise ValueError("Schedule missing 'week' column required to filter games by week.")
    
    week_games = schedule[schedule['week'] == week].copy()
    
    if week_games.empty:
        raise ValueError(f"No games found for week {week}.")
    
    week_games = week_games.sort_values('game_date').reset_index(drop=True)
    print(f"\nFound {len(week_games)} games for week {week}:")
    for _, row in week_games.iterrows():
        print(f"  {row['game_date'].date()} - {row['away_team']} @ {row['home_team']}")
    
    base_cols = ['home_team', 'away_team', 'game_date', 'season', 'week']
    optional_cols = [
        col for col in ['start_time', 'gametime', 'game_time', 'game_time_utc']
        if col in week_games.columns
    ]
    return week_games[base_cols + optional_cols].copy()


def team_has_data(team: str, game_date: pd.Timestamp, completed_games: pd.DataFrame) -> bool:
    """Check whether the team has at least one current-season completed game before the given date."""
    games_before = completed_games[
        (completed_games['game_date'] < game_date) &
        ((completed_games['home_team'] == team) | (completed_games['away_team'] == team))
    ]
    return len(games_before) > 0


def predict_games(games: pd.DataFrame,
                  schedule: pd.DataFrame,
                  completed_games: pd.DataFrame,
                  play_by_play: Optional[pd.DataFrame]) -> List[dict]:
    """Predict a batch of games, skipping any without sufficient data."""
    predictor = GamePredictor('NFL', MODEL_VERSION)
    
    predictions = []
    for _, game in games.iterrows():
        game_date = pd.to_datetime(game['game_date'])
        home_team = game['home_team']
        away_team = game['away_team']
        
        if not team_has_data(home_team, game_date, completed_games):
            print(f"\nSkipping {away_team} @ {home_team} ({game_date.date()}): "
                  f"No completed games for {home_team} before this date.")
            continue
        if not team_has_data(away_team, game_date, completed_games):
            print(f"\nSkipping {away_team} @ {home_team} ({game_date.date()}): "
                  f"No completed games for {away_team} before this date.")
            continue
        
        game_df = pd.DataFrame([game])
        result = predictor.predict(game_df, schedule, play_by_play=play_by_play)
        predictions.append(result)
    
    return predictions


def display_predictions(predictions: List[dict], week: int) -> None:
    """Pretty-print prediction results."""
    if not predictions:
        print("\nNo predictions generated.")
        return
    
    print("\n" + "=" * 80)
    print(f"WEEK {week} PREDICTIONS")
    print("=" * 80)
    
    for pred in predictions:
        print(f"\n{pred['away_team']} @ {pred['home_team']}  |  {pred['game_date']}")
        print(f"  Spread: {pred['predicted_spread']:.2f} ({pred['spread_interpretation']})")
        print(f"  Win Probabilities: Home {pred['home_win_probability']:.1%} | Away {pred['away_win_probability']:.1%}")
        print(f"  Predicted Winner: {pred['predicted_winner']} (Confidence {pred['confidence']:.1%})")
        print(f"  Model win prob: {pred.get('home_win_prob_from_model', float('nan')):.1%}")
        print(f"  From spread: {pred.get('win_prob_from_spread', float('nan')):.1%}")
        if pred.get('model_disagreement', 0) > 0.15:
            print(f"  ⚠️  Disagreement: {pred['model_disagreement']:.1%}")
    
    print("\n" + "=" * 80)


def write_viz_data(predictions: List[dict], games_df: pd.DataFrame, schedule_df: pd.DataFrame, week: int) -> None:
    """Export predictions to parquet file for notebook visualization."""
    if not predictions:
        print("No predictions to export; skipping viz data write.")
        return
    
    # Convert predictions to DataFrame
    preds_df = pd.DataFrame(predictions)
    preds_df['game_date'] = pd.to_datetime(preds_df['game_date'])
    
    # Merge with full schedule to get game_id (schedule has game_id, filtered games_df might not)
    schedule_copy = schedule_df.copy()
    schedule_copy['game_date'] = pd.to_datetime(schedule_copy['game_date'])
    
    # Check if game_id exists in schedule
    if 'game_id' not in schedule_copy.columns:
        print("Warning: schedule DataFrame missing 'game_id' column. Cannot export viz data.")
        return
    
    # Merge on home_team, away_team, and game_date
    merged = preds_df.merge(
        schedule_copy[['home_team', 'away_team', 'game_date', 'game_id']].drop_duplicates(
            subset=['home_team', 'away_team', 'game_date']
        ),
        on=['home_team', 'away_team', 'game_date'],
        how='left'
    )
    
    if merged['game_id'].isna().any():
        print("Warning: Some predictions missing game_id. These will be excluded from export.")
        merged = merged[merged['game_id'].notna()]
    
    if merged.empty:
        print("No predictions with valid game_id to export.")
        return
    
    # Prepare export DataFrame with required columns
    export_df = pd.DataFrame({
        'game_id': merged['game_id'],
        'my_spread': merged['predicted_spread'],
        'my_home_win_prob': merged['home_win_probability']
    })
    
    # Create exports directory if it doesn't exist
    export_dir = "notebooks/exports"
    os.makedirs(export_dir, exist_ok=True)
    
    # Write to parquet
    output_path = f"{export_dir}/week{week}_python_preds.parquet"
    export_df.to_parquet(output_path, index=False)
    print(f"\nExported {len(export_df)} predictions to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict NFL games for a specific week and optionally push results to Supabase."
    )
    parser.add_argument(
        "--week",
        type=int,
        required=True,
        help="Week number to predict (e.g., 11, 12, 13).",
    )
    parser.add_argument(
        "--season",
        type=int,
        default=2025,
        help="NFL season to score (default: 2025).",
    )
    parser.add_argument(
        "--push-to-db",
        action="store_true",
        help="Persist predictions to Supabase using env credentials.",
    )
    parser.add_argument(
        "--write-viz-data",
        action="store_true",
        help="Export predictions to parquet file for notebook visualization.",
    )
    return parser.parse_args()


def load_supabase_credentials() -> dict:
    """Ensure required Supabase env vars are available before writes."""
    load_dotenv()
    creds = {
        'url': os.getenv('SUPABASE_URL'),
        'service_role_key': os.getenv('SUPABASE_SERVICE_ROLE_KEY'),
        'db_password': os.getenv('supabaseDBpass'),
        'db_host': os.getenv('SUPABASE_DB_HOST'),
        'db_name': os.getenv('SUPABASE_DB_NAME', 'postgres'),
        'db_port': os.getenv('SUPABASE_DB_PORT', '5432'),
        'db_user': os.getenv('SUPABASE_DB_USER', 'postgres')
    }
    missing = [key for key in ('url', 'service_role_key', 'db_password') if not creds[key]]
    if missing:
        raise EnvironmentError(
            f"Missing Supabase environment variables: {', '.join(missing)}"
        )
    try:
        creds['db_port'] = int(creds['db_port'])
    except ValueError as exc:
        raise EnvironmentError("SUPABASE_DB_PORT must be an integer") from exc
    os.environ.setdefault('SUPABASE_DB_PASSWORD', creds['db_password'])
    return creds


def _infer_game_time_utc(row: pd.Series) -> pd.Timestamp:
    """Best-effort conversion to UTC for Supabase games table."""
    start_time = row.get('start_time')
    if pd.notna(start_time):
        ts = pd.to_datetime(start_time, utc=True, errors='coerce')
        if pd.notna(ts):
            return ts
    
    gametime = row.get('gametime')
    if pd.notna(gametime):
        try:
            base_date = pd.to_datetime(row['game_date']).date()
            combined = f"{base_date} {gametime}"
            ts = pd.to_datetime(combined, utc=True, errors='coerce')
            if pd.notna(ts):
                return ts
        except Exception:
            pass
    
    existing = row.get('game_time_utc')
    if pd.notna(existing):
        ts = pd.to_datetime(existing, utc=True, errors='coerce')
        if pd.notna(ts):
            return ts
    
    return pd.to_datetime(row['game_date'], utc=True, errors='coerce')


def prepare_games_for_supabase(
    games_df: pd.DataFrame,
    schedule_df: pd.DataFrame
) -> pd.DataFrame:
    """Enrich games with metadata required for Supabase upserts."""
    if games_df.empty:
        return games_df
    
    games = games_df.copy()
    games['game_date'] = pd.to_datetime(games['game_date'])
    games['league'] = 'NFL'
    if 'week' not in games.columns:
        games['week'] = pd.NA
    games['week'] = games['week'].astype('Int64')
    
    schedule = schedule_df.copy()
    schedule['game_date'] = pd.to_datetime(schedule['game_date'])
    join_cols = ['home_team', 'away_team', 'game_date', 'season']
    extra_cols = [
        col for col in ['start_time', 'gametime', 'game_time_utc']
        if col in schedule.columns and col not in games.columns
    ]
    
    if extra_cols:
        schedule_subset = schedule[join_cols + extra_cols].drop_duplicates(subset=join_cols)
        games = games.merge(schedule_subset, on=join_cols, how='left', suffixes=('', '_schedule'))
    
    games['game_time_utc'] = games.apply(_infer_game_time_utc, axis=1)
    games['game_time_utc'] = pd.to_datetime(games['game_time_utc'], utc=True, errors='coerce')
    games['season'] = games['season'].astype(int)
    return games


def push_predictions_to_supabase(
    predictions: List[dict],
    games_df: pd.DataFrame,
    schedule_df: pd.DataFrame
) -> None:
    """Push predictions and related games to Supabase tables."""
    if not predictions:
        print("No predictions to push; skipping Supabase write.")
        return
    
    creds = load_supabase_credentials()
    games_for_db = prepare_games_for_supabase(games_df, schedule_df)
    if games_for_db.empty:
        print("No game metadata available for Supabase; skipping write.")
        return
    
    conn = create_pg_connection(
        supabase_url=creds['url'],
        password=creds['db_password'],
        host_override=creds.get('db_host'),
        port=creds['db_port'],
        database=creds['db_name'],
        user=creds['db_user']
    )
    try:
        game_id_map = upsert_games_pg(conn, games_for_db)
        predictions_df = pd.DataFrame(predictions).rename(columns={
            'predicted_spread': 'my_spread',
            'home_win_probability': 'my_home_win_prob'
        })
        prediction_payload = predictions_df[
            ['home_team', 'away_team', 'game_date', 'my_spread', 'my_home_win_prob']
        ]
        insert_predictions_pg(conn, prediction_payload, game_id_map)
        print(f"Pushed {len(prediction_payload)} predictions to Supabase.")
    finally:
        conn.close()


def create_pg_connection(
    supabase_url: str,
    password: str,
    host_override: Optional[str] = None,
    port: int = 5432,
    database: str = 'postgres',
    user: str = 'postgres'
) -> psycopg.Connection:
    """Build and open a psycopg connection to the Supabase Postgres instance."""
    if host_override:
        db_host = host_override
        host_source = "override (SUPABASE_DB_HOST)"
    else:
        if not supabase_url:
            raise ValueError("Supabase URL is required to derive the database host.")
        parsed = urlparse(supabase_url)
        host = parsed.netloc.split(':')[0]
        if not host:
            raise ValueError("Supabase URL is missing a hostname.")
        project_ref = host.split('.')[0]
        if not project_ref:
            raise ValueError("Unable to infer Supabase project ref from SUPABASE_URL.")
        db_host = f"db.{project_ref}.supabase.co"
        host_source = f"derived (project_ref={project_ref})"
    print(
        f"[Supabase] Attempting connection -> host={db_host} ({host_source}), "
        f"port={port}, database={database}, user={user}"
    )
    conn_str = (
        f"postgresql://{user}:{password}@{db_host}:{int(port)}/{database}?sslmode=require"
    )
    try:
        return psycopg.connect(conn_str)
    except OSError as err:
        raise ConnectionError(
            f"Unable to resolve or reach Supabase host '{db_host}' on port {port}. "
            "Set SUPABASE_DB_HOST to override the default if your project uses a custom domain."
        ) from err


def game_map_key(home_team: str, away_team: str, game_date: pd.Timestamp) -> Tuple[str, str, date]:
    """Consistent tuple key for mapping games to prediction rows."""
    date_only = pd.to_datetime(game_date).date()
    return (home_team, away_team, date_only)


def upsert_games_pg(conn: psycopg.Connection, games_df: pd.DataFrame) -> dict:
    """Insert or update games and return a map for downstream prediction inserts."""
    if games_df.empty:
        return {}
    
    select_sql = """
        select id
        from games
        where league = %s
          and season = %s
          and home_team = %s
          and away_team = %s
          and game_time_utc::date = %s
        order by game_time_utc desc
        limit 1
    """
    update_sql = """
        update games
        set game_time_utc = %s,
            week = coalesce(%s, week)
        where id = %s
    """
    insert_sql = """
        insert into games (league, season, week, game_time_utc, home_team, away_team)
        values (%s, %s, %s, %s, %s, %s)
        returning id
    """
    
    game_id_map = {}
    with conn.cursor() as cur:
        for _, row in games_df.iterrows():
            has_game_time = 'game_time_utc' in row.index
            game_time = pd.to_datetime(row['game_time_utc'], utc=True, errors='coerce') if has_game_time and pd.notna(row['game_time_utc']) else None
            if pd.isna(game_time):
                game_time = pd.to_datetime(row['game_date'], utc=True)
            game_time = game_time.to_pydatetime()
            game_date = pd.to_datetime(row['game_date']).date()
            week_val = int(row['week']) if 'week' in row.index and pd.notna(row['week']) else None
            params = (
                row['league'],
                int(row['season']),
                row['home_team'],
                row['away_team'],
                game_date
            )
            cur.execute(select_sql, params, prepare=False)
            existing = cur.fetchone()
            if existing:
                game_id = existing[0]
                cur.execute(update_sql, (game_time, week_val, game_id), prepare=False)
            else:
                cur.execute(
                    insert_sql,
                    (row['league'], int(row['season']), week_val, game_time, row['home_team'], row['away_team']),
                    prepare=False
                )
                game_id = cur.fetchone()[0]
            key = game_map_key(row['home_team'], row['away_team'], game_date)
            game_id_map[key] = game_id
    conn.commit()
    return game_id_map


def insert_predictions_pg(
    conn: psycopg.Connection,
    predictions_df: pd.DataFrame,
    game_id_map: dict
) -> None:
    """Insert prediction rows linked to previously upserted games."""
    if predictions_df.empty:
        print("No predictions frame to push; skipping prediction insert.")
        return
    
    insert_sql = """
        insert into model_predictions
            (game_id, model_name, model_version, my_spread, my_home_win_prob)
        values (%s, %s, %s, %s, %s)
    """
    inserted = 0
    with conn.cursor() as cur:
        for _, row in predictions_df.iterrows():
            key = game_map_key(row['home_team'], row['away_team'], row['game_date'])
            game_id = game_id_map.get(key)
            if not game_id:
                continue
            spread = float(row['my_spread']) if pd.notna(row['my_spread']) else None
            win_prob = float(row['my_home_win_prob']) if pd.notna(row['my_home_win_prob']) else None
            cur.execute(
                insert_sql,
                (
                    game_id,
                    'sports_edge_weekly',
                    MODEL_VERSION,
                    spread,
                    win_prob
                )
            )
            inserted += 1
    conn.commit()
    print(f"Inserted {inserted} rows into model_predictions.")


def main():
    args = parse_args()
    season = args.season
    week = args.week
    
    try:
        schedule_df = load_season_schedule(season)
    except Exception as err:
        print(f"ERROR: {err}")
        sys.exit(1)
    
    try:
        completed_games = filter_completed_games(schedule_df)
    except Exception as err:
        print(f"ERROR: {err}")
        sys.exit(1)
    
    print(f"\nUsing {season} season data:")
    print(f"  Total games: {len(schedule_df)}")
    print(f"  Date range: {schedule_df['game_date'].min().date()} to {schedule_df['game_date'].max().date()}")

    play_by_play = load_pbp([season])
    
    try:
        week_games = collect_week_games(schedule_df, week)
    except Exception as err:
        print(f"ERROR: {err}")
        sys.exit(1)
    
    predictions = predict_games(week_games, schedule_df, completed_games, play_by_play)
    display_predictions(predictions, week)
    
    if args.write_viz_data:
        try:
            write_viz_data(predictions, week_games, schedule_df, week)
        except Exception as err:
            print(f"ERROR writing viz data: {err}")
            sys.exit(1)
    
    if args.push_to_db:
        try:
            push_predictions_to_supabase(predictions, week_games, schedule_df)
        except Exception as err:
            print(f"ERROR pushing to Supabase: {err}")
            sys.exit(1)


if __name__ == "__main__":
    main()

