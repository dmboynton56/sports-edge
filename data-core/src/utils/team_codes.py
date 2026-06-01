"""Team code canonicalization helpers shared by sync scripts."""

from __future__ import annotations

from typing import Optional


NBA_TEAM_NAME_TO_ABBR = {
    "atlanta hawks": "ATL",
    "boston celtics": "BOS",
    "brooklyn nets": "BKN",
    "charlotte hornets": "CHA",
    "chicago bulls": "CHI",
    "cleveland cavaliers": "CLE",
    "dallas mavericks": "DAL",
    "denver nuggets": "DEN",
    "detroit pistons": "DET",
    "golden state warriors": "GSW",
    "houston rockets": "HOU",
    "indiana pacers": "IND",
    "la clippers": "LAC",
    "los angeles clippers": "LAC",
    "la lakers": "LAL",
    "los angeles lakers": "LAL",
    "memphis grizzlies": "MEM",
    "miami heat": "MIA",
    "milwaukee bucks": "MIL",
    "minnesota timberwolves": "MIN",
    "new orleans pelicans": "NOP",
    "new york knicks": "NYK",
    "oklahoma city thunder": "OKC",
    "orlando magic": "ORL",
    "philadelphia 76ers": "PHI",
    "phoenix suns": "PHX",
    "portland trail blazers": "POR",
    "sacramento kings": "SAC",
    "san antonio spurs": "SAS",
    "toronto raptors": "TOR",
    "utah jazz": "UTA",
    "washington wizards": "WAS",
}

NBA_SCHEDULE_TO_ODDS_MAP = {
    "GS": "GSW",
    "NO": "NOP",
    "NY": "NYK",
    "SA": "SAS",
    "UTAH": "UTA",
    "WSH": "WAS",
    "PHO": "PHX",
    "BRK": "BKN",
    "BRX": "BKN",
}

NFL_TEAM_NAME_TO_ABBR = {
    "arizona cardinals": "ARI",
    "atlanta falcons": "ATL",
    "baltimore ravens": "BAL",
    "buffalo bills": "BUF",
    "carolina panthers": "CAR",
    "chicago bears": "CHI",
    "cincinnati bengals": "CIN",
    "cleveland browns": "CLE",
    "dallas cowboys": "DAL",
    "denver broncos": "DEN",
    "detroit lions": "DET",
    "green bay packers": "GB",
    "houston texans": "HOU",
    "indianapolis colts": "IND",
    "jacksonville jaguars": "JAX",
    "kansas city chiefs": "KC",
    "las vegas raiders": "LV",
    "los angeles chargers": "LAC",
    "la chargers": "LAC",
    "los angeles rams": "LAR",
    "la rams": "LAR",
    "miami dolphins": "MIA",
    "minnesota vikings": "MIN",
    "new england patriots": "NE",
    "new orleans saints": "NO",
    "new york giants": "NYG",
    "ny giants": "NYG",
    "new york jets": "NYJ",
    "ny jets": "NYJ",
    "philadelphia eagles": "PHI",
    "pittsburgh steelers": "PIT",
    "san francisco 49ers": "SF",
    "seattle seahawks": "SEA",
    "tampa bay buccaneers": "TB",
    "tennessee titans": "TEN",
    "washington commanders": "WAS",
}

NFL_ALIAS_TO_ABBR = {
    "ARZ": "ARI",
    "JAC": "JAX",
    "WSH": "WAS",
}

MLB_TEAM_NAME_TO_ABBR = {
    "arizona diamondbacks": "AZ",
    "athletics": "ATH",
    "oakland athletics": "ATH",
    "atlanta braves": "ATL",
    "baltimore orioles": "BAL",
    "boston red sox": "BOS",
    "chicago cubs": "CHC",
    "chicago white sox": "CWS",
    "cincinnati reds": "CIN",
    "cleveland guardians": "CLE",
    "colorado rockies": "COL",
    "detroit tigers": "DET",
    "houston astros": "HOU",
    "kansas city royals": "KC",
    "los angeles angels": "LAA",
    "la angels": "LAA",
    "los angeles dodgers": "LAD",
    "la dodgers": "LAD",
    "miami marlins": "MIA",
    "milwaukee brewers": "MIL",
    "minnesota twins": "MIN",
    "new york mets": "NYM",
    "ny mets": "NYM",
    "new york yankees": "NYY",
    "ny yankees": "NYY",
    "philadelphia phillies": "PHI",
    "pittsburgh pirates": "PIT",
    "san diego padres": "SD",
    "san francisco giants": "SF",
    "seattle mariners": "SEA",
    "st louis cardinals": "STL",
    "st. louis cardinals": "STL",
    "tampa bay rays": "TB",
    "texas rangers": "TEX",
    "toronto blue jays": "TOR",
    "washington nationals": "WSH",
}

MLB_ALIAS_TO_ABBR = {
    "ARI": "AZ",
    "ATH": "ATH",
    "OAK": "ATH",
    "CHW": "CWS",
    "KCR": "KC",
    "SDP": "SD",
    "SFG": "SF",
    "TBR": "TB",
    "WSN": "WSH",
    "WAS": "WSH",
}


def canonical_nba_abbr(code_or_name: object) -> Optional[str]:
    """Return the Odds API-style NBA abbreviation for a schedule/database code."""
    if not isinstance(code_or_name, str):
        return None

    text = code_or_name.strip()
    if not text:
        return None

    name_match = NBA_TEAM_NAME_TO_ABBR.get(text.lower())
    if name_match:
        return name_match

    token = text.upper()
    return NBA_SCHEDULE_TO_ODDS_MAP.get(token, token)


def canonical_nfl_abbr(code_or_name: object) -> Optional[str]:
    """Return the canonical NFL abbreviation for a schedule/database code."""
    if not isinstance(code_or_name, str):
        return None

    text = code_or_name.strip()
    if not text:
        return None

    name_match = NFL_TEAM_NAME_TO_ABBR.get(text.lower())
    if name_match:
        return name_match

    token = text.upper()
    return NFL_ALIAS_TO_ABBR.get(token, token)


def canonical_mlb_abbr(code_or_name: object) -> Optional[str]:
    """Return the canonical MLB abbreviation used by Supabase score matching."""
    if not isinstance(code_or_name, str):
        return None

    text = code_or_name.strip()
    if not text:
        return None

    name_match = MLB_TEAM_NAME_TO_ABBR.get(text.lower())
    if name_match:
        return name_match

    token = text.upper()
    return MLB_ALIAS_TO_ABBR.get(token, token)


def canonical_team_abbr(league: object, code_or_name: object) -> Optional[str]:
    """Return a canonical team abbreviation for supported daily refresh leagues."""
    if not isinstance(league, str):
        return None

    league_key = league.upper()
    if league_key == "NBA":
        return canonical_nba_abbr(code_or_name)
    if league_key == "NFL":
        return canonical_nfl_abbr(code_or_name)
    if league_key == "MLB":
        return canonical_mlb_abbr(code_or_name)
    return code_or_name.strip().upper() if isinstance(code_or_name, str) else None
