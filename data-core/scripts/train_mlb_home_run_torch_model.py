#!/usr/bin/env python3
"""Train a PyTorch MLB batter home-run probability experiment.

This consumes the same leakage-aware training rows produced by
train_mlb_home_run_model.py, then compares a GPU-friendly wide/deep model
against the current random-forest artifact and the stored baseline probability.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_DATASET = ROOT / "notebooks" / "cache" / "mlb_home_run_training_rows.csv"
DEFAULT_RF_MODEL = ROOT / "models" / "mlb_hr_model_v1.joblib"
DEFAULT_MODEL = ROOT / "models" / "mlb_hr_torch_model_v1.pt"
DEFAULT_METRICS = ROOT / "models" / "mlb_hr_torch_model_v1_metrics.json"
MODEL_VERSION = "mlb-hr-torch-v1"
RF_MODEL_VERSION = "mlb-hr-v1"
BASE_FEATURE_COLUMNS = [
    "batter_pa_lag",
    "batter_games_lag",
    "batter_hr_per_pa",
    "batter_recent_pa_28",
    "batter_recent_hr_per_pa_28",
    "batter_rate_over_league",
    "recent_rate_over_league",
    "pitcher_bf_lag",
    "pitcher_hr_factor",
    "venue_factor",
    "league_hr_per_pa",
    "lineup_slot",
    "expected_pa",
    "is_home",
]
CONTINUOUS_COLUMNS = BASE_FEATURE_COLUMNS + ["baseline_probability", "heuristic_probability"]
CATEGORICAL_COLUMNS = ["player_id", "opposing_starter_id", "team_id", "opponent_id"]
OPTIONAL_CONTINUOUS_COLUMNS = [
    "batter_bats_left",
    "batter_bats_right",
    "batter_switch_hitter",
    "pitcher_throws_left",
    "pitcher_throws_right",
    "known_handedness_matchup",
    "batter_platoon_advantage",
    "same_side_matchup",
    "batter_statcast_pitches_lag",
    "batter_statcast_bbe_lag",
    "batter_avg_ev_lag",
    "batter_avg_la_lag",
    "batter_hard_hit_rate_lag",
    "batter_barrel_proxy_rate_lag",
    "batter_sweet_spot_rate_lag",
    "batter_fb_ld_rate_lag",
    "batter_statcast_hr_per_pitch_lag",
    "batter_fastball_share_lag",
    "batter_breaking_share_lag",
    "batter_offspeed_share_lag",
    "pitcher_statcast_pitches_lag",
    "pitcher_statcast_bbe_lag",
    "pitcher_avg_ev_lag",
    "pitcher_avg_la_lag",
    "pitcher_hard_hit_rate_lag",
    "pitcher_barrel_proxy_rate_lag",
    "pitcher_sweet_spot_rate_lag",
    "pitcher_fb_ld_rate_lag",
    "pitcher_statcast_hr_per_pitch_lag",
    "pitcher_fastball_share_lag",
    "pitcher_breaking_share_lag",
    "pitcher_offspeed_share_lag",
    "statcast_feature_ready",
]
OPTIONAL_CATEGORICAL_COLUMNS = ["batter_bat_side", "pitcher_throw_hand"]


def _date_arg(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _load_common_deps() -> None:
    global joblib, np, pd, SimpleImputer, StandardScaler, brier_score_loss, log_loss, roc_auc_score
    try:
        import joblib
        import numpy as np
        import pandas as pd
        from sklearn.impute import SimpleImputer
        from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
        from sklearn.preprocessing import StandardScaler
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing Python model dependencies. Install data-core/requirements.txt in the "
            "project environment, then rerun this script."
        ) from exc


def _load_torch() -> Any:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyTorch is not installed in this Python environment. Install a CUDA-enabled "
            "torch build, then rerun this script."
        ) from exc
    return torch, nn, DataLoader, TensorDataset


def _clip_prob(values: Any) -> Any:
    return np.clip(np.asarray(values, dtype=float), 1e-6, 1 - 1e-6)


def _top_k_hit_rate(frame: pd.DataFrame, probs: np.ndarray, k: int) -> float:
    tmp = frame[["game_date", "actual_home_run"]].copy()
    tmp["probability"] = probs
    top = (
        tmp.sort_values(["game_date", "probability"], ascending=[True, False])
        .groupby("game_date", group_keys=False)
        .head(k)
    )
    return float(top["actual_home_run"].mean()) if len(top) else float("nan")


def _calibration(frame: pd.DataFrame, probs: np.ndarray, bins: int = 10) -> list[dict[str, Any]]:
    tmp = pd.DataFrame({"actual": frame["actual_home_run"].astype(int), "probability": probs})
    try:
        tmp["bucket"] = pd.qcut(tmp["probability"], q=bins, duplicates="drop")
    except ValueError:
        return []
    rows = []
    for bucket, group in tmp.groupby("bucket", observed=True):
        rows.append(
            {
                "bucket": str(bucket),
                "rows": int(len(group)),
                "avg_probability": float(group["probability"].mean()),
                "actual_rate": float(group["actual"].mean()),
            }
        )
    return rows


def _evaluate(frame: pd.DataFrame, probs: np.ndarray) -> dict[str, Any]:
    y = frame["actual_home_run"].astype(int).to_numpy()
    clipped = _clip_prob(probs)
    metrics: dict[str, Any] = {
        "rows": int(len(frame)),
        "positive_rate": float(y.mean()),
        "brier": float(brier_score_loss(y, clipped)),
        "log_loss": float(log_loss(y, clipped, labels=[0, 1])),
        "top_10_hit_rate": _top_k_hit_rate(frame, clipped, 10),
        "top_25_hit_rate": _top_k_hit_rate(frame, clipped, 25),
        "calibration": _calibration(frame, clipped),
    }
    if len(np.unique(y)) > 1:
        metrics["auc"] = float(roc_auc_score(y, clipped))
    return metrics


def _evaluate_named(frame: pd.DataFrame, probabilities: dict[str, np.ndarray]) -> dict[str, Any]:
    return {name: _evaluate(frame, probs) for name, probs in probabilities.items()}


def _best_blend_weight(actual: pd.Series, primary_probs: np.ndarray, secondary_probs: np.ndarray) -> tuple[float, float]:
    y = actual.astype(int)
    best_weight = 1.0
    best_loss = math.inf
    for weight in np.linspace(0.0, 1.0, 51):
        probs = _clip_prob((weight * primary_probs) + ((1.0 - weight) * secondary_probs))
        loss = float(log_loss(y, probs, labels=[0, 1]))
        if loss < best_loss:
            best_loss = loss
            best_weight = float(weight)
    return best_weight, best_loss


def _encode_categories(
    train: pd.DataFrame,
    test: pd.DataFrame,
    categorical_columns: list[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, dict[str, int]], list[int]]:
    train_arrays = []
    test_arrays = []
    mappings: dict[str, dict[str, int]] = {}
    cardinalities = []
    for col in categorical_columns:
        train_values = train[col].fillna("__missing__").astype(str)
        unique_values = sorted(train_values.unique().tolist())
        mapping = {value: idx + 1 for idx, value in enumerate(unique_values)}
        mappings[col] = mapping
        train_arrays.append(train_values.map(mapping).fillna(0).to_numpy(dtype=np.int64))
        test_arrays.append(test[col].fillna("__missing__").astype(str).map(mapping).fillna(0).to_numpy(dtype=np.int64))
        cardinalities.append(len(mapping) + 1)
    return (
        np.vstack(train_arrays).T,
        np.vstack(test_arrays).T,
        mappings,
        cardinalities,
    )


def _prepare_continuous(
    train: Any,
    test: Any,
    continuous_columns: list[str],
) -> tuple[Any, Any, Any, Any]:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_imputed = imputer.fit_transform(train[continuous_columns])
    test_imputed = imputer.transform(test[continuous_columns])
    train_scaled = scaler.fit_transform(train_imputed).astype(np.float32)
    test_scaled = scaler.transform(test_imputed).astype(np.float32)
    return train_scaled, test_scaled, imputer, scaler


def _predict(
    torch: Any,
    model: Any,
    cat_x: np.ndarray,
    cont_x: np.ndarray,
    *,
    device: Any,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    probs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(cat_x), batch_size):
            cat = torch.tensor(cat_x[start : start + batch_size], dtype=torch.long, device=device)
            cont = torch.tensor(cont_x[start : start + batch_size], dtype=torch.float32, device=device)
            logits = model(cat, cont)
            probs.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(probs)


def _time_validation_split(train: Any, validation_fraction: float = 0.2) -> tuple[Any, Any]:
    split_date = train["game_date"].quantile(1.0 - validation_fraction).normalize()
    fit = train[train["game_date"] < split_date].copy()
    validation = train[train["game_date"] >= split_date].copy()
    if len(fit) < 500 or len(validation) < 250 or validation["actual_home_run"].nunique() < 2:
        cutoff = max(1, int(len(train) * (1.0 - validation_fraction)))
        fit = train.iloc[:cutoff].copy()
        validation = train.iloc[cutoff:].copy()
    if len(fit) < 500 or len(validation) < 250 or validation["actual_home_run"].nunique() < 2:
        raise SystemExit("Inner train/validation split is too small for PyTorch early stopping.")
    return fit, validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PyTorch MLB HR model from existing training rows.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--test-start-date", type=_date_arg, default=date(2026, 5, 15))
    parser.add_argument("--rf-model", type=Path, default=DEFAULT_RF_MODEL)
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--metrics-out", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--model-version", default=MODEL_VERSION)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _load_common_deps()
    if not args.dataset.exists():
        raise SystemExit(
            f"Missing training rows: {args.dataset}. Run scripts/train_mlb_home_run_model.py first."
        )

    torch, nn, DataLoader, TensorDataset = _load_torch()
    from src.models.mlb_hr_torch_model import MlbHrWideDeepNN

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("Requested CUDA, but torch.cuda.is_available() is false.")

    rows = pd.read_csv(args.dataset)
    rows["game_date"] = pd.to_datetime(rows["game_date"])
    continuous_columns = CONTINUOUS_COLUMNS + [
        col for col in OPTIONAL_CONTINUOUS_COLUMNS if col in rows.columns
    ]
    categorical_columns = CATEGORICAL_COLUMNS + [
        col for col in OPTIONAL_CATEGORICAL_COLUMNS if col in rows.columns
    ]
    required = {"actual_home_run", "game_date", *continuous_columns, *categorical_columns}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise SystemExit(f"Training rows are missing required columns: {missing}")

    rows = rows.sort_values(["game_date", "game_pk", "player_id"]).reset_index(drop=True)
    split_ts = pd.Timestamp(args.test_start_date)
    train = rows[rows["game_date"] < split_ts].copy()
    test = rows[rows["game_date"] >= split_ts].copy()
    if len(train) < 500 or len(test) < 250:
        split_date = rows["game_date"].quantile(0.8).normalize()
        train = rows[rows["game_date"] < split_date].copy()
        test = rows[rows["game_date"] >= split_date].copy()
        print(f"Requested split was too small; using fallback split date {split_date.date()}.", flush=True)
    if train["actual_home_run"].nunique() < 2 or test["actual_home_run"].nunique() < 2:
        raise SystemExit("Train/test split does not contain both target classes.")

    train_cat, test_cat, category_mappings, cardinalities = _encode_categories(train, test, categorical_columns)
    train_cont, test_cont, imputer, scaler = _prepare_continuous(train, test, continuous_columns)
    fit, validation = _time_validation_split(train)
    fit_pos = train.index.get_indexer(fit.index)
    validation_pos = train.index.get_indexer(validation.index)
    fit_cat = train_cat[fit_pos]
    validation_cat = train_cat[validation_pos]
    fit_cont = train_cont[fit_pos]
    validation_cont = train_cont[validation_pos]
    y_fit = fit["actual_home_run"].astype(np.float32).to_numpy()

    dataset = TensorDataset(
        torch.tensor(fit_cat, dtype=torch.long),
        torch.tensor(fit_cont, dtype=torch.float32),
        torch.tensor(y_fit, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=device.type == "cuda")

    model = MlbHrWideDeepNN(
        cardinalities,
        train_cont.shape[1],
        embedding_dim=args.embedding_dim,
    ).to(device)
    prior = float(np.clip(fit["actual_home_run"].mean(), 1e-4, 1 - 1e-4))
    with torch.no_grad():
        model.output.bias.fill_(math.log(prior / (1.0 - prior)))
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_state = copy.deepcopy(model.state_dict())
    best_loss = math.inf
    stale_epochs = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for cat, cont, target in loader:
            cat = cat.to(device)
            cont = cont.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(cat, cont), target)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(target)

        train_loss = total_loss / len(dataset)
        validation_probs = _predict(
            torch,
            model,
            validation_cat,
            validation_cont,
            device=device,
            batch_size=args.batch_size,
        )
        val_loss = float(log_loss(validation["actual_home_run"].astype(int), _clip_prob(validation_probs), labels=[0, 1]))
        history.append({"epoch": float(epoch), "train_loss": train_loss, "validation_log_loss": val_loss})
        print(f"Epoch {epoch}: train_loss={train_loss:.5f} validation_log_loss={val_loss:.5f}", flush=True)
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= args.patience:
            print(f"Early stopping after {epoch} epochs.", flush=True)
            break

    model.load_state_dict(best_state)
    train_probs = _predict(torch, model, train_cat, train_cont, device=device, batch_size=args.batch_size)
    test_probs = _predict(torch, model, test_cat, test_cont, device=device, batch_size=args.batch_size)
    validation_probs = _predict(
        torch,
        model,
        validation_cat,
        validation_cont,
        device=device,
        batch_size=args.batch_size,
    )
    heuristic_validation = validation["heuristic_probability"].to_numpy(dtype=float)
    heuristic_test = test["heuristic_probability"].to_numpy(dtype=float)
    blend_weight, blend_validation_log_loss = _best_blend_weight(
        validation["actual_home_run"],
        validation_probs,
        heuristic_validation,
    )
    test_blend_probs = (blend_weight * test_probs) + ((1.0 - blend_weight) * heuristic_test)

    comparison_probs: dict[str, np.ndarray] = {
        "baseline_probability": test["baseline_probability"].to_numpy(dtype=float),
        "heuristic_probability": heuristic_test,
        args.model_version: test_probs,
        f"{args.model_version}_heuristic_blend": test_blend_probs,
    }
    rf_status = "missing"
    if args.rf_model.exists():
        try:
            rf_artifact = joblib.load(args.rf_model)
            rf_columns = rf_artifact.get("feature_columns", CONTINUOUS_COLUMNS)
            rf_probs = rf_artifact["model"].predict_proba(test[rf_columns])[:, 1]
            comparison_probs[rf_artifact.get("model_version", RF_MODEL_VERSION)] = rf_probs
            rf_status = "evaluated"
        except Exception as exc:  # noqa: BLE001
            rf_status = f"failed: {type(exc).__name__}: {exc}"

    metrics = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "model_version": args.model_version,
        "estimator": "pytorch_wide_deep",
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "training_rows": int(len(rows)),
        "training_window": {
            "dataset": str(args.dataset),
            "test_start_date": args.test_start_date.isoformat(),
            "train_rows": int(len(train)),
            "fit_rows": int(len(fit)),
            "validation_rows": int(len(validation)),
            "test_rows": int(len(test)),
        },
        "leakage_controls": [
            "training rows are generated chronologically by train_mlb_home_run_model.py",
            "only pregame lagged batter, pitcher, venue, and league features are used",
            "time split is preserved for PyTorch and baseline comparisons",
        ],
        "feature_columns": continuous_columns,
        "categorical_columns": categorical_columns,
        "train": _evaluate(train, train_probs),
        "validation": _evaluate(validation, validation_probs),
        "test": _evaluate(test, test_probs),
        "blend": {
            "name": f"{args.model_version}_heuristic_blend",
            "pytorch_weight": blend_weight,
            "heuristic_weight": 1.0 - blend_weight,
            "validation_log_loss": blend_validation_log_loss,
            "test": _evaluate(test, test_blend_probs),
        },
        "same_split_comparison": _evaluate_named(test, comparison_probs),
        "rf_artifact_status": rf_status,
        "epoch_history": history,
    }

    artifact = {
        "state_dict": best_state,
        "model_version": args.model_version,
        "estimator": "pytorch_wide_deep",
        "continuous_columns": continuous_columns,
        "categorical_columns": categorical_columns,
        "categorical_cardinalities": cardinalities,
        "category_mappings": category_mappings,
        "imputer": imputer,
        "scaler": scaler,
        "metrics": metrics,
        "model_kwargs": {
            "embedding_dim": args.embedding_dim,
            "num_continuous": int(train_cont.shape[1]),
            "categorical_cardinalities": cardinalities,
        },
    }
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, args.model_out)
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    print(f"Wrote PyTorch artifact to {args.model_out}", flush=True)
    print(f"Wrote metrics to {args.metrics_out}", flush=True)
    print(
        "Test metrics: "
        f"Brier={metrics['test']['brier']:.4f}, "
        f"log_loss={metrics['test']['log_loss']:.4f}, "
        f"AUC={metrics['test'].get('auc', float('nan')):.4f}, "
        f"top10_hit_rate={metrics['test']['top_10_hit_rate']:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
