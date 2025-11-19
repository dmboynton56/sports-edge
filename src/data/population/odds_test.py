from src.data.odds_fetcher import fetch_odds
df = fetch_odds(
    league="NFL",
    date="2025-11-18",
    markets="spreads",
    regions="us",
    bookmakers="fanduel"
)
print(df[["home_team", "away_team", "commence_time", "book", "line"]])
