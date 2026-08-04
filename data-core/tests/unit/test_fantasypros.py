from src.fantasy.fantasypros import normalize_consensus_rows, normalize_projection_rows
from src.fantasy.fantasypros import FantasyProsClient


def test_adp_consensus_rank_fallback_is_preserved():
    rows = normalize_consensus_rows(
        {
            "last_updated": "8/03",
            "players": [
                {
                    "player_id": 123,
                    "player_name": "Example Player",
                    "player_team_id": "BUF",
                    "player_position_id": "RB",
                    "rank_ecr": 7,
                    "rank_std": "1.2",
                }
            ],
        }
    )
    assert rows[0]["adp"] == 7
    assert rows[0]["consensus_rank"] == 7
    assert rows[0]["consensus_std"] == 1.2


def test_projection_normalization_handles_documented_stat_payload():
    rows = normalize_projection_rows(
        {
            "season": 2026,
            "week": 0,
            "players": [
                {
                    "fpid": 123,
                    "name": "Example Player",
                    "position_id": "RB",
                    "team_id": "BUF",
                    "stats": {"rush_yds": 100.5, "rec_rec": 4},
                }
            ],
        }
    )
    assert rows[0]["player_id"] == 123
    assert rows[0]["stats"]["rush_yds"] == 100.5
    assert rows[0]["week"] == 0


def test_consensus_inputs_match_v2_query_names(monkeypatch):
    captured = {}

    class Response:
        ok = True

        def json(self):
            return {"players": []}

    def fake_get(url, *, headers, params, timeout):
        captured.update(url=url, headers=headers, params=params, timeout=timeout)
        return Response()

    monkeypatch.setattr("src.fantasy.fantasypros.requests.get", fake_get)
    client = FantasyProsClient(api_key="test-key")
    client.consensus_rankings(
        season=2026,
        position="RB",
        scoring="HALF",
        ranking_type="ADP",
        week=0,
        experts=True,
        filters=["12", "34"],
    )

    assert captured["url"].endswith("/nfl/2026/consensus-rankings")
    assert captured["headers"]["x-api-key"] == "test-key"
    assert captured["params"] == {
        "position": "RB",
        "scoring": "HALF",
        "type": "ADP",
        "week": 0,
        "experts": "true",
        "filters": "12:34",
    }
