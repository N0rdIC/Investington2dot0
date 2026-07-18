"""Train and freeze the 3-class barrier model, with per-side calibration.

WHY 3 CLASSES
-------------
One softmax over {SHORT_WIN, NEITHER, LONG_WIN} instead of two independent
binary models. The three outcomes are mutually exclusive by construction (a path
that reaches +6% before -3% crossed +3% on the way, which is the short's stop),
so the softmax constraint p_short + p_neither + p_long = 1 is not an
approximation -- it is the truth. Two separate binary models violated it in ~40%
of predictions.

WHY CALIBRATION IS STILL NEEDED
-------------------------------
A class probability is NOT a success rate. Regularisation and the multiclass
normalisation both distort it. We fit isotonic regression on OUT-OF-SAMPLE
predictions, separately per side, to map

    class probability  ->  observed success rate

and only the calibrated value is compared against breakeven P*.

No class weighting is applied: the three classes are near-balanced, and
weighting would decalibrate the very quantity we need to read as a probability.

Outputs:
    ../model/model3.json
    ../model/meta.json
    ../model/calibration.json
    ../model/training_history.json
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from qcs.config import Config
from qcs.data import load_yahoo
from qcs.features import build_features, cross_sectional_zscore
from qcs.labels3 import joint_barrier_labels, class_distribution, verify_exhaustive
from evaluate3 import evaluate3, print_report3
from evaluate import append_history, HISTORY

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
MODEL_DIR.mkdir(exist_ok=True)

UP, DOWN = 0.06, 0.03
VOL_CAP_COLS = {"rvol_252", "rvol_60", "rvol_ratio", "rvol_ratio_long",
                "vol_of_vol", "mkt_stress", "mkt_stress_z"}


def make_fitter(cfg, feature_cols):
    m = cfg.model
    fw = np.ones(len(feature_cols))
    for i, c in enumerate(feature_cols):
        if c in VOL_CAP_COLS:
            fw[i] = cfg.model_vol_cap_weight

    def _fit(Xtr, ytr):
        mdl = xgb.XGBClassifier(
            n_estimators=m.n_estimators, max_depth=m.max_depth,
            learning_rate=m.learning_rate, subsample=m.subsample,
            colsample_bytree=m.colsample_bytree,
            min_child_weight=m.min_child_weight,
            reg_lambda=m.reg_lambda, random_state=m.random_state,
            objective="multi:softprob", num_class=3, eval_metric="mlogloss",
            feature_weights=fw, n_jobs=4, verbosity=0,
        )
        # only the label-overlap correction; NO class weighting (see docstring)
        mdl.fit(Xtr.to_numpy(), ytr.to_numpy().astype(int),
                sample_weight=np.full(len(ytr), 1.0 / cfg.label.horizon))
        return mdl

    return _fit


def main():
    cfg = Config()
    print(f"Downloading {len(cfg.universe)} names...")
    px = load_yahoo(cfg.universe, cfg.start, cfg.end)
    fx = load_yahoo(cfg.sector_etfs, cfg.start, cfg.end)
    sx = load_yahoo(cfg.stress_tickers, cfg.start, cfg.end)

    close, volume = px["close"], px["volume"]
    ok = close.notna().sum() > 400
    close, volume = close.loc[:, ok], volume.loc[:, ok]
    print(f"  {close.shape[1]} names x {close.shape[0]} days")

    v = verify_exhaustive(close, UP, DOWN, cfg.label.horizon)
    print(f"  exclusivity: overlap={v['overlap']}  exclusive={v['exclusive']}")
    if not v["exclusive"]:
        raise RuntimeError("class definition broken: long/short outcomes overlap")

    print("Building features...")
    X = cross_sectional_zscore(
        build_features(close, volume, fx["close"], stress_close=sx["close"]))
    y = joint_barrier_labels(close, UP, DOWN, cfg.label.horizon)

    idx = X.index.intersection(y.index)
    X, y = X.loc[idx].sort_index(), y.loc[idx].sort_index()
    mask = y.notna() & (X.notna().sum(axis=1) >= 0.5 * X.shape[1])
    Xf, yf = X[mask], y[mask].astype(int)

    dist = class_distribution(yf)
    p_star = (DOWN + cfg.costs.round_trip) / (UP + DOWN)
    print(f"  samples={len(Xf)}  classes={dist}  P*={p_star:.4f}")

    fit = make_fitter(cfg, list(Xf.columns))

    print("\nEvaluating out-of-sample (purged walk-forward)...")
    metrics = evaluate3(Xf, yf, cfg, fit, p_star, UP, DOWN)

    prev = None
    if HISTORY.exists():
        try:
            h = json.loads(HISTORY.read_text())
            prev = h[-1].get("metrics") if h else None
        except (json.JSONDecodeError, IndexError, AttributeError):
            prev = None
    print_report3(metrics, prev)

    print(f"\nFitting final model on all {len(Xf)} samples...")
    model = fit(Xf, yf)
    model.save_model(str(MODEL_DIR / "model3.json"))

    meta = {
        "model_type": "3class_calibrated",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "features": list(Xf.columns),
        "universe": list(close.columns),
        "up": UP, "down": DOWN, "horizon": cfg.label.horizon,
        "p_star": p_star,
        "class_distribution": dist,
        "threshold_long": metrics.get("threshold_long"),
        "threshold_short": metrics.get("threshold_short"),
        "train_start": str(close.index[0].date()),
        "train_end": str(close.index[-1].date()),
        "n_samples": int(len(Xf)),
    }
    (MODEL_DIR / "meta.json").write_text(json.dumps(meta, indent=2))

    append_history({
        "trained_at": meta["trained_at"],
        "train_start": meta["train_start"], "train_end": meta["train_end"],
        "n_samples": meta["n_samples"], "n_names": len(meta["universe"]),
        "n_features": len(meta["features"]),
        "metrics": metrics,
    })

    for old in ("model_long.json", "model_short.json"):
        p = MODEL_DIR / old
        if p.exists():
            p.unlink(); print(f"  removed stale {old}")

    print(f"\nSaved model3.json.  thresholds: long={metrics.get('threshold_long')}"
          f"  short={metrics.get('threshold_short')}")


if __name__ == "__main__":
    main()
