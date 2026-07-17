"""Train and freeze the barrier models. Run manually (workflow_dispatch).

Unlike the backtest -- which trains 6 walk-forward models to measure honest
out-of-sample skill -- this fits ONE model per side on ALL available history,
because for live trading you want the model to use every scrap of data before
predicting tomorrow. The walk-forward backtest already told us the signal is
real out-of-sample; this step is about deployment, not validation.

Outputs (committed by the workflow):
    ../model/model_long.json
    ../model/model_short.json
    ../model/meta.json

The daily job then loads these frozen files. Retraining is manual so you control
exactly when the model changes -- no silent drift between what you validated and
what trades.
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
from qcs.labels import triple_barrier_labels

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
MODEL_DIR.mkdir(exist_ok=True)

UP, DOWN = 0.06, 0.03


def _fit(X, y, cfg, vol_cap_cols):
    y = y.astype(int)
    spw = float((y == 0).sum()) / max(1, (y == 1).sum())
    fw = np.ones(X.shape[1])
    for i, c in enumerate(X.columns):
        if c in vol_cap_cols:
            fw[i] = cfg.model_vol_cap_weight
    m = cfg.model
    model = xgb.XGBClassifier(
        n_estimators=m.n_estimators, max_depth=m.max_depth,
        learning_rate=m.learning_rate, subsample=m.subsample,
        colsample_bytree=m.colsample_bytree, min_child_weight=m.min_child_weight,
        reg_lambda=m.reg_lambda, random_state=m.random_state,
        objective="binary:logistic", eval_metric="logloss",
        scale_pos_weight=spw, feature_weights=fw, n_jobs=4, verbosity=0,
    )
    w = np.full(len(y), 1.0 / cfg.label.horizon)
    model.fit(X.to_numpy(), y.to_numpy(), sample_weight=w)
    return model


def main():
    cfg = Config()
    print(f"Downloading {len(cfg.universe)} names...")
    px = load_yahoo(cfg.universe, cfg.start, cfg.end)
    fx = load_yahoo(cfg.sector_etfs, cfg.start, cfg.end)
    sx = load_yahoo(cfg.stress_tickers, cfg.start, cfg.end)

    close, volume = px["close"], px["volume"]
    ok = close.notna().sum() > 400
    close, volume = close.loc[:, ok], volume.loc[:, ok]
    factor_close, stress_close = fx["close"], sx["close"]

    print("Building features...")
    X = cross_sectional_zscore(
        build_features(close, volume, factor_close, stress_close=stress_close))

    y_long = triple_barrier_labels(close, UP, DOWN, cfg.label.horizon, "long")
    y_short = triple_barrier_labels(close, UP, DOWN, cfg.label.horizon, "short")

    idx = X.index.intersection(y_long.index)
    X = X.loc[idx].sort_index()
    yl = y_long.loc[idx].sort_index()
    ys = y_short.loc[idx].sort_index()

    # drop rows without a label or with too few features
    mask = yl.notna() & (X.notna().sum(axis=1) >= 0.5 * X.shape[1])
    Xf, ylf, ysf = X[mask], yl[mask], ys[mask]

    vol_cap = {"rvol_252", "rvol_60", "rvol_ratio", "rvol_ratio_long",
               "vol_of_vol", "mkt_stress", "mkt_stress_z"}

    print(f"Training on {len(Xf)} samples, {Xf.shape[1]} features...")
    ml = _fit(Xf, ylf, cfg, vol_cap)
    ms = _fit(Xf, ysf, cfg, vol_cap)

    ml.save_model(str(MODEL_DIR / "model_long.json"))
    ms.save_model(str(MODEL_DIR / "model_short.json"))

    p_star = (DOWN + cfg.costs.round_trip) / (UP + DOWN)
    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "features": list(Xf.columns),
        "universe": list(close.columns),
        "up": UP, "down": DOWN, "horizon": cfg.label.horizon,
        "p_star": p_star,
        "train_start": str(close.index[0].date()),
        "train_end": str(close.index[-1].date()),
        "n_samples": int(len(Xf)),
        "base_rate_long": float(ylf.mean()),
        "base_rate_short": float(ysf.mean()),
    }
    (MODEL_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved models. p_star={p_star:.3f}, "
          f"base rate long={meta['base_rate_long']:.3f}")


if __name__ == "__main__":
    main()
