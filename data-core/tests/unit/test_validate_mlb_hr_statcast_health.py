from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import validate_mlb_hr_statcast_health as statcast_health  # noqa: E402


def _write_health(tmp_path: Path, health: dict) -> Path:
    path = tmp_path / "mlb_home_runs.json"
    path.write_text(json.dumps({"statcastHealth": health}), encoding="utf-8")
    return path


def test_warn_only_allows_low_statcast_coverage(tmp_path, monkeypatch, capsys) -> None:
    path = _write_health(
        tmp_path,
        {
            "enabled": True,
            "artifactLoaded": True,
            "coverage": 0.0,
            "readyRows": 0,
            "totalRows": 120,
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_mlb_hr_statcast_health.py",
            "--warn-only",
            "--min-coverage",
            "0.50",
            str(path),
        ],
    )

    statcast_health.main()

    assert "WARNING: Statcast coverage 0.000 is below floor 0.500." in capsys.readouterr().out


def test_strict_mode_fails_low_statcast_coverage(tmp_path, monkeypatch) -> None:
    path = _write_health(
        tmp_path,
        {
            "enabled": True,
            "artifactLoaded": True,
            "coverage": 0.0,
            "readyRows": 0,
            "totalRows": 120,
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_mlb_hr_statcast_health.py",
            "--min-coverage",
            "0.50",
            str(path),
        ],
    )

    with pytest.raises(SystemExit, match="Statcast coverage 0.000 is below floor 0.500."):
        statcast_health.main()


def test_warn_only_allows_missing_artifact(tmp_path, monkeypatch, capsys) -> None:
    path = _write_health(
        tmp_path,
        {
            "enabled": True,
            "artifactLoaded": False,
            "artifactError": "model unavailable",
            "coverage": 0.0,
            "readyRows": 0,
            "totalRows": 120,
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["validate_mlb_hr_statcast_health.py", "--warn-only", str(path)],
    )

    statcast_health.main()

    assert "WARNING: Statcast blend artifact did not load: model unavailable" in capsys.readouterr().out
