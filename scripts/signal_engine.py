"""Signal engine: turn the pretrained 3-class model + fresh prices into signals.

Shared by train.py and daily.py so the features computed live are identical to
those the model trained on -- the most common source of train/serve skew.

Model files in ../model/ :
    model3.json  - one XGBoost multiclass model over
                   {0: short_win, 1: neither, 2: long_win}
    meta.json    - feature list, barrier params, p_star

Because the model is a single softmax, its outputs satisfy
    p_short + p_neither + p_long = 1
so p_long and p_short can never both be high. The old two-model setup violated
this in ~40% of predictions.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from qcs.features import build_features, cross_sectional_zscore

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
SHORT_WIN, NEITHER, LONG_WIN = 0, 1, 2


def compute_features(close, volume, factor_close, stress_close=None):
    """Build the exact feature panel the model expects, cross-sectionally z-scored."""
    panel = build_features(close, volume, factor_close, stress_close=stress_close)
    return cross_sectional_zscore(panel)


def latest_scores(close, volume, factor_close, stress_close, model, meta):
    """Today's class probabilities per ticker.

    Returns a DataFrame indexed by ticker with p_short / p_neither / p_long,
    plus the derived `tilt` and `magnitude`.
    """
    X = compute_features(close, volume, factor_close, stress_close)
    last_date = X.index.get_level_values("date").max()
    Xt = X.xs(last_date, level="date")

    feat_cols = meta["features"]
    Xt = Xt.reindex(columns=feat_cols)
    keep = Xt.notna().sum(axis=1) >= 0.5 * len(feat_cols)
    Xt = Xt[keep]

    P = model.predict_proba(Xt.to_numpy())
    out = pd.DataFrame(
        {"p_short": P[:, SHORT_WIN], "p_neither": P[:, NEITHER],
         "p_long": P[:, LONG_WIN]},
        index=Xt.index,
    )
    denom = (out["p_long"] + out["p_short"]).replace(0, np.nan)
    out["tilt"] = (out["p_long"] - out["p_short"]) / denom
    out["magnitude"] = out["p_long"] + out["p_short"]
    out["date"] = str(pd.Timestamp(last_date).date())
    return out.sort_values("p_long", ascending=False)


def pick_signals(scores, meta, n_side=5, margin=0.02, min_tilt=0.10):
    """Convert probabilities into concrete BUY / SHORT signals.

    Two gates, both must pass:

    1. EXPECTANCY GATE: p clears breakeven P* = (L+c)/(G+L) by `margin`.
       This is the original P*G - (1-P)*L - c condition.

    2. DIRECTION GATE: |tilt| >= min_tilt, where
           tilt = (p_long - p_short) / (p_long + p_short)

    With the 3-class model the probabilities are already mutually consistent, so
    gate 2 no longer corrects a contradiction -- it now does what it should:
    require a clear directional call rather than a marginal one. A name at
    (0.40, 0.38) has real barrier-touch probability but no direction; a name at
    (0.40, 0.12) is a genuine call. Set min_tilt=0.0 to disable.
    """
    p_star = meta["p_star"]
    thr = p_star + margin
    s = scores

    longs = s[(s["p_long"] > thr) & (s["tilt"] >= min_tilt)].nlargest(n_side, "tilt")
    shorts = s[(s["p_short"] > thr) & (s["tilt"] <= -min_tilt)].nsmallest(n_side, "tilt")
    shorts = shorts[~shorts.index.isin(longs.index)]

    k = min(len(longs), len(shorts)) if (len(longs) and len(shorts)) else 0
    if k > 0:
        longs, shorts = longs.head(k), shorts.head(k)

    sig = []
    for tkr, row in longs.iterrows():
        sig.append({"ticker": tkr, "side": "long",
                    "p": float(row["p_long"]), "p_opp": float(row["p_short"]),
                    "p_neither": float(row["p_neither"]),
                    "tilt": round(float(row["tilt"]), 3)})
    for tkr, row in shorts.iterrows():
        sig.append({"ticker": tkr, "side": "short",
                    "p": float(row["p_short"]), "p_opp": float(row["p_long"]),
                    "p_neither": float(row["p_neither"]),
                    "tilt": round(float(row["tilt"]), 3)})
    return sig


def load_models():
    """Load the 3-class model. Returns (model, meta)."""
    import xgboost as xgb

    meta_path = MODEL_DIR / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            "No meta.json. Run the retrain-model workflow first.")
    meta = json.loads(meta_path.read_text())

    m3 = MODEL_DIR / "model3.json"
    if not m3.exists():
        raise FileNotFoundError(
            "model3.json not found. The model is now a single 3-class model -- "
            "run the retrain-model workflow to regenerate it.")

    model = xgb.XGBClassifier()
    model.load_model(str(m3))
    return model, meta
