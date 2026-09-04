from src.utils.injury_loader import _all_uuid_game_ids, load_injury_impacts_from_supabase


def test_all_uuid_game_ids_accepts_supabase_ids():
    game_id = "e0e5ba7b-42bc-46b2-b1fd-9cbb78a1f68d"

    assert _all_uuid_game_ids([game_id]) == [game_id]


def test_all_uuid_game_ids_rejects_warehouse_schedule_keys():
    assert _all_uuid_game_ids(["2026_01_NE_SEA"]) is None


def test_all_uuid_game_ids_rejects_mixed_identity_domains():
    assert _all_uuid_game_ids(
        [
            "e0e5ba7b-42bc-46b2-b1fd-9cbb78a1f68d",
            "2026_01_NE_SEA",
        ]
    ) is None


def _clear_supabase_pg_env(monkeypatch):
    for name in (
        "SUPABASE_URL",
        "SUPABASE_DB_HOST",
        "supabaseDBpass",
        "SUPABASE_DB_PASSWORD",
        "SUPABASE_DB_PORT",
        "SUPABASE_DB_NAME",
        "SUPABASE_DB_USER",
    ):
        monkeypatch.delenv(name, raising=False)


def test_load_injury_impacts_skips_when_url_missing(monkeypatch, capsys):
    _clear_supabase_pg_env(monkeypatch)
    monkeypatch.setenv("supabaseDBpass", "secret")

    def boom(**kwargs):
        raise AssertionError("create_pg_connection should not run without a host")

    monkeypatch.setattr("src.utils.injury_loader.create_pg_connection", boom)

    frame = load_injury_impacts_from_supabase("NFL", game_ids=["2026_01_NE_SEA"])

    assert frame.empty
    captured = capsys.readouterr().out
    assert "WARNING" in captured
    assert "SUPABASE_URL or SUPABASE_DB_HOST" in captured
    assert "continue without injuries" in captured


def test_load_injury_impacts_skips_when_password_missing(monkeypatch, capsys):
    _clear_supabase_pg_env(monkeypatch)
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")

    def boom(**kwargs):
        raise AssertionError("create_pg_connection should not run without a password")

    monkeypatch.setattr("src.utils.injury_loader.create_pg_connection", boom)

    frame = load_injury_impacts_from_supabase("NBA")

    assert frame.empty
    captured = capsys.readouterr().out
    assert "WARNING" in captured
    assert "supabaseDBpass or SUPABASE_DB_PASSWORD" in captured


def test_load_injury_impacts_skips_when_create_pg_connection_raises_valueerror(monkeypatch, capsys):
    _clear_supabase_pg_env(monkeypatch)
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("supabaseDBpass", "secret")

    def boom(**kwargs):
        raise ValueError("Missing Supabase Postgres host: set SUPABASE_DB_HOST or SUPABASE_URL")

    monkeypatch.setattr("src.utils.injury_loader.create_pg_connection", boom)

    frame = load_injury_impacts_from_supabase("NFL")

    assert frame.empty
    captured = capsys.readouterr().out
    assert "WARNING" in captured
    assert "continue without injuries" in captured
