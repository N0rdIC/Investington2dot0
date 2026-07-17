"""Signal engine: turn a pretrained model + fresh prices into today's signals.

Shared by train.py (which fits and saves the models) and daily.py (which loads
them and scores the latest bar). Keeping one code path guarantees the features
computed live are identical to those the model trained on -- the single most
common source of train/serve skew.

The model files live in ../model/ :
    model_long.json   - XGBoost classifier for P(+G before -L)
    model_short.json  - XGBoost classifier for the mirror (short) barrier
    meta.json         - feature list, barrier params, training window, p_star

This module imports the feature code from the qcs package, which must be on the
PYTHONPATH (the workflows copy the qcs/ folder next to these scripts).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from qcs.features import build_features, cross_sectional_zscore

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"


def compute_features(close, volume, factor_close, stress_close=None):
    """Build the exact feature panel the model expects, cross-sectionally z-scored."""
    panel = build_features(close, volume, factor_close, stress_close=stress_close)
    return cross_sectional_zscore(panel)


def latest_scores(close, volume, factor_close, stress_close, models, meta):
    """Return today's predicted P(event) for long and short books.

    Uses only the most recent date for which a full feature row can be built.
    Returns a DataFrame indexed by ticker with columns p_long, p_short.
    """
    X = compute_features(close, volume, factor_close, stress_close)

    # newest date present in the feature panel
    last_date = X.index.get_level_values("date").max()
    Xt = X.xs(last_date, level="date")

    # align columns to the trained feature order; missing -> NaN (xgb handles it)
    feat_cols = meta["features"]
    Xt = Xt.reindex(columns=feat_cols)

    # keep rows with at least half the features present
    keep = Xt.notna().sum(axis=1) >= 0.5 * len(feat_cols)
    Xt = Xt[keep]

    out = pd.DataFrame(index=Xt.index)
    out["p_long"] = models["long"].predict_proba(Xt.to_numpy())[:, 1]
    out["p_short"] = models["short"].predict_proba(Xt.to_numpy())[:, 1]
    out["date"] = str(pd.Timestamp(last_date).date())
    return out.sort_values("p_long", ascending=False)


def pick_signals(scores, meta, n_side=5, margin=0.02):
    """Convert probabilities into concrete BUY / SHORT signals.

    A name is a long candidate if p_long clears the breakeven P* by `margin`,
    a short candidate if p_short does. We take the top `n_side` of each and, to
    keep the book roughly drift-neutral, balance the counts.
    """
    p_star = meta["p_star"]
    thr = p_star + margin

    longs = scores[scores["p_long"] > thr].nlargest(n_side, "p_long")
    shorts = scores[scores["p_short"] > thr].nlargest(n_side, "p_short")
    shorts = shorts[~shorts.index.isin(longs.index)]

    k = min(len(longs), len(shorts)) if (len(longs) and len(shorts)) else 0
    if k > 0:
        longs = longs.head(k)
        shorts = shorts.head(k)
    else:
        # if one side is empty, still allow the other (paper trading tolerates it)
        pass

    sig = []
    for tkr, row in longs.iterrows():
        sig.append({"ticker": tkr, "side": "long", "p": float(row["p_long"])})
    for tkr, row in shorts.iterrows():
        sig.append({"ticker": tkr, "side": "short", "p": float(row["p_short"])})
    return sig


def load_models():
    import xgboost as xgb

    meta = json.loads((MODEL_DIR / "meta.json").read_text())
    ml = xgb.XGBClassifier()
    ml.load_model(str(MODEL_DIR / "model_long.json"))
    ms = xgb.XGBClassifier()
    ms.load_model(str(MODEL_DIR / "model_short.json"))
    return {"long": ml, "short": ms}, meta
