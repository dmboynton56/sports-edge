
from nba_api.stats.endpoints import leaguegamefinder
import pandas as pd

def check_season(season_str):
    print(f"Checking {season_str}...")
    
    # helper
    def get_count(season_type):
        finder = leaguegamefinder.LeagueGameFinder(
            season_nullable=season_str,
            league_id_nullable='00',
            season_type_nullable=season_type
        )
        df = finder.get_data_frames()[0]
        return len(df)

    reg = get_count('Regular Season')
    print(f"Regular Season: {reg}")
    
    # check other types
    pre = get_count('Pre Season')
    print(f"Pre Season: {pre}")
    
    # IST?
    # some potential values: 'In-Season Tournament' doesn't exist as a type in classical API usually, 
    # but let's check wildcards or None if possible.
    
    # Try getting ALL by not passing season_type?
    # The class sets a default if we don't pass it.
    # We can try to manually override the parameter dict if needed, but let's see if we can pass something else.
    # What if we pass Set(SeasonType) etc?
    
    # It seems LeagueGameFinder might not support "All" easily without iterating.
    
    # What about teamgamelogs?
    from nba_api.stats.static import teams
    celts = teams.find_teams_by_full_name('Celtics')[0]
    from nba_api.stats.endpoints import teamgamelog
    tgl = teamgamelog.TeamGameLog(team_id=celts['id'], season='2025-26')
    df_tgl = tgl.get_data_frames()[0]
    print(f"Celtics (2025-26) Games: {len(df_tgl)}")

check_season("2025-26")
