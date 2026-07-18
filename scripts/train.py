"""Train and freeze the 3-class barrier model. Run manually (workflow_dispatch).

ONE multiclass model replaces the two independent binary ones. The softmax
normalises across {short_win, neither, long_win}, so

    p_short + p_neither + p_long = 1

is guaranteed. With two separate binary models, ~40% of predictions had
p_long + p_short > 1 -- mathematically impossible, since the two outcomes are
mutually exclusive (hitting +6% before -3% means the path crossed +3% on the
way up, which is the short's stop). That inconsistency is now inexpressible
rather than filtered out after the fact.

Unlike the backtest -- which trains walk-forward models to measure honest
out-of-sample skill -- this fits ONE model on ALL available history, because
for live trading you want every scrap of data before predicting tomorrow.

Outputs (committed by the workflow):
    ../model/model3.json
    ../model/meta.json
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
from qcs.labels3 import (joint_barrier_labels, class_distribution,
                         verify_exhaustive)

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
MODEL_DIR.mkdir(exist_ok=True)

UP, DOWN = 0.06, 0.03
VOL_CAP_COLS = {"rvol_252", "rvol_60", "rvol_ratio", "rvol_ratio_long",
                "vol_of_vol", "mkt_stress", "mkt_stress_z"}


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
    print(f"  {close.shape[1]} names x {close.shape[0]} days")

    # sanity: the three classes must be mutually exclusive or nothing works
    v = verify_exhaustive(close, UP, DOWN, cfg.label.horizon)
    print(f"  exclusivity check: overlap={v['overlap']} exclusive={v['exclusive']}")
    if not v["exclusive"]:
        raise RuntimeError("class definition broken: long/short outcomes overlap")

    print("Building features...")
    X = cross_sectional_zscore(
        build_features(close, volume, factor_close, stress_close=stress_close))
    y = joint_barrier_labels(close, UP, DOWN, cfg.label.horizon)

    idx = X.index.intersection(y.index)
    X, y = X.loc[idx].sort_index(), y.loc[idx].sort_index()
    mask = y.notna() & (X.notna().sum(axis=1) >= 0.5 * X.shape[1])
    Xf, yf = X[mask], y[mask].astype(int)

    dist = class_distribution(yf)
    print(f"  classes: {dist}")

    # SAMPLE WEIGHTS: only the label-overlap correction, NOT class balancing.
    #
    # Why no class weights: the three classes are already nearly balanced
    # (~29 / 37 / 34), so inverse-frequency weighting corrects nothing -- but it
    # DOES distort calibration, because it makes the model predict under a
    # uniform 1/3 prior instead of the true base rates. That matters here in a
    # way it doesn't for pure ranking: we compare p_long against an ABSOLUTE
    # threshold P* = (L+c)/(G+L). Decalibrated probabilities make that
    # comparison meaningless. So we keep the probabilities honest and only
    # down-weight for the fact that consecutive 5-day labels overlap.
    sw = np.full(len(yf), 1.0 / cfg.label.horizon)

    fw = np.ones(Xf.shape[1])
    for i, c in enumerate(Xf.columns):
        if c in VOL_CAP_COLS:
            fw[i] = cfg.model_vol_cap_weight

    m = cfg.model
    print(f"Training 3-class model on {len(Xf)} samples, {Xf.shape[1]} features...")
    model = xgb.XGBClassifier(
        n_estimators=m.n_estimators, max_depth=m.max_depth,
        learning_rate=m.learning_rate, subsample=m.subsample,
        colsample_bytree=m.colsample_bytree, min_child_weight=m.min_child_weight,
        reg_lambda=m.reg_lambda, random_state=m.random_state,
        objective="multi:softprob", num_class=3, eval_metric="mlogloss",
        feature_weights=fw, n_jobs=4, verbosity=0,
    )
    model.fit(Xf.to_numpy(), yf.to_numpy(), sample_weight=sw)
    model.save_model(str(MODEL_DIR / "model3.json"))

    p_star = (DOWN + cfg.costs.round_trip) / (UP + DOWN)
    meta = {
        "model_type": "3class",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "features": list(Xf.columns),
        "universe": list(close.columns),
        "up": UP, "down": DOWN, "horizon": cfg.label.horizon,
        "p_star": p_star,
        "train_start": str(close.index[0].date()),
        "train_end": str(close.index[-1].date()),
        "n_samples": int(len(Xf)),
        "class_distribution": dist,
    }
    (MODEL_DIR / "meta.json").write_text(json.dumps(meta, indent=2))

    # remove stale two-model artifacts so daily.py can't silently load them
    for old in ("model_long.json", "model_short.json"):
        p = MODEL_DIR / old
        if p.exists():
            p.unlink()
            print(f"  removed stale {old}")

    print(f"Saved model3.json. p_star={p_star:.3f}")


if __name__ == "__main__":
    main()
