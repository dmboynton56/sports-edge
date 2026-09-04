from src.utils.injury_loader import _all_uuid_game_ids


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
