def get_base_performance_query(limit: int = 1000) -> str:
    """
    Query to pull round-by-round scores, Strokes Gained metrics, and basic stats.
    Assumes standard table structures based on GCP setup.
    """
    limit_clause = f"LIMIT {limit}" if limit else ""
    return f"""
        SELECT 
            t1.player_name,
            t1.tournament_id,
            t1.date,
            t1.round,
            t1.score,
            t1.sg_putt,
            t1.sg_arg,
            t1.sg_app,
            t1.sg_ott,
            t1.sg_t2g,
            t1.sg_total,
            t1.birdies,
            t1.bogeys,
            t1.fairways_hit
        FROM `sports_edge_raw.raw_pga_player_rounds` t1
        ORDER BY t1.date DESC, t1.player_name
        {limit_clause}
    """

def get_course_environmental_query(limit: int = 1000) -> str:
    """
    Query to pull venue information, course difficulty, and weather data.
    """
    limit_clause = f"LIMIT {limit}" if limit else ""
    return f"""
        SELECT 
            c.course_id,
            c.course_name,
            c.par,
            c.yardage,
            c.historical_difficulty,
            c.driving_distance_importance,
            w.tournament_id,
            w.round,
            w.wind_speed,
            w.temperature,
            w.precipitation
        FROM `sports_edge_raw.raw_pga_courses` c
        LEFT JOIN `sports_edge_raw.raw_pga_weather` w 
            ON c.course_id = w.course_id
        {limit_clause}
    """

def get_player_baseline_query(limit: int = 1000) -> str:
    """
    Query to pull historical rolling averages (True Skill baseline).
    """
    limit_clause = f"LIMIT {limit}" if limit else ""
    return f"""
        SELECT 
            player_name,
            date,
            AVG(sg_total) OVER (
                PARTITION BY player_name 
                ORDER BY date 
                ROWS BETWEEN 50 PRECEDING AND CURRENT ROW
            ) as sg_total_50_round_avg,
            AVG(sg_app) OVER (
                PARTITION BY player_name 
                ORDER BY date 
                ROWS BETWEEN 50 PRECEDING AND CURRENT ROW
            ) as sg_app_50_round_avg
        FROM `sports_edge_raw.raw_pga_player_rounds`
        ORDER BY date DESC, player_name
        {limit_clause}
    """
