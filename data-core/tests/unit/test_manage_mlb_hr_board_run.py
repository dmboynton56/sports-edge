from argparse import Namespace

import pytest

from scripts import manage_mlb_hr_board_run as board_run


def args() -> Namespace:
    return Namespace(run_key="afternoon-101", slate_date="2026-08-28")


def test_nonempty_publish_never_checks_previous_board(monkeypatch):
    monkeypatch.setattr(
        board_run,
        "_previous_nonempty_board",
        lambda *_: pytest.fail("non-empty publishes must not invoke the empty-overwrite guard"),
    )
    board_run._refuse_unsafe_empty_overwrite(object(), args(), [{"game_id": "MLB_1"}])


def test_empty_publish_is_allowed_without_a_previous_nonempty_board(monkeypatch):
    monkeypatch.setattr(board_run, "_previous_nonempty_board", lambda *_: None)
    board_run._refuse_unsafe_empty_overwrite(object(), args(), [])


def test_empty_publish_refuses_to_replace_a_live_board(monkeypatch):
    monkeypatch.setattr(board_run, "_previous_nonempty_board", lambda *_: ("morning-100", ["MLB_1"]))
    monkeypatch.setattr(board_run, "_official_schedule_confirms_slate_over", lambda *_: False)

    with pytest.raises(RuntimeError, match="morning-100 remains live"):
        board_run._refuse_unsafe_empty_overwrite(object(), args(), [])


def test_empty_publish_is_allowed_after_official_slate_completion(monkeypatch):
    monkeypatch.setattr(board_run, "_previous_nonempty_board", lambda *_: ("morning-100", ["MLB_1"]))
    monkeypatch.setattr(board_run, "_official_schedule_confirms_slate_over", lambda *_: True)
    board_run._refuse_unsafe_empty_overwrite(object(), args(), [])


def test_empty_publish_fails_closed_when_completion_check_errors(monkeypatch):
    monkeypatch.setattr(board_run, "_previous_nonempty_board", lambda *_: ("morning-100", ["MLB_1"]))

    def unavailable(*_):
        raise TimeoutError("MLB schedule unavailable")

    monkeypatch.setattr(board_run, "_official_schedule_confirms_slate_over", unavailable)
    with pytest.raises(RuntimeError, match="completion could not be verified"):
        board_run._refuse_unsafe_empty_overwrite(object(), args(), [])
