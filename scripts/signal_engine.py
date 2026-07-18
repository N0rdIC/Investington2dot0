"""Signal engine: 3-class model -> calibrated success rates -> signals.

TWO STEPS BETWEEN MODEL AND TRADE, both essential:

  1. CALIBRATION. The softmax output is a CLASS PROBABILITY, not a success rate.
     Isotonic maps (fit out-of-sample per side during training) convert it:
         p_long  -> P(actually hits +G before -L)
         p_short -> P(actually hits -G before +L)
     Only the calibrated number may be compared against breakeven P*.

  2. THRESHOLD. Each side has its own threshold, chosen by the training sweep as
     the lowest one whose MEASURED precision clears P* by a safety margin with
     enough samples behind it. If a side has no validated threshold, that side
     emits nothing -- sitting out is a correct outcome, not a failure.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from qcs.features import build_features, cross_sectional_zscore
from calibration import load_calibration, apply_calibration

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
SHORT_WIN, NEITHER, LONG_WIN = 0, 1, 2


def compute_features(close, volume, factor_close, stress_close=None):
    panel = build_features(close, volume, factor_close, stress_close=stress_close)
    return cross_sectional_zscore(panel)


def latest_scores(close, volume, factor_close, stress_close, model, meta):
    X = compute_features(close, volume, factor_close, stress_close)
    last_date = X.index.get_level_values("date").max()
    Xt = X.xs(last_date, level="date").reindex(columns=meta["features"])
    Xt = Xt[Xt.notna().sum(axis=1) >= 0.5 * len(meta["features"])]

    P = model.predict_proba(Xt.to_numpy())
    out = pd.DataFrame({
        "p_short_raw": P[:, SHORT_WIN],
        "p_neither": P[:, NEITHER],
        "p_long_raw": P[:, LONG_WIN],
    }, index=Xt.index)

    cal = load_calibration()
    out["p_long"] = apply_calibration(out["p_long_raw"], cal.get("long", {}))
    out["p_short"] = apply_calibration(out["p_short_raw"], cal.get("short", {}))
    out["calibrated"] = bool(cal.get("long"))

    # tilt on RAW scores: they share one softmax, so their difference is already
    # on a common scale. Per-side calibration is monotone but not comparable.
    denom = (out["p_long_raw"] + out["p_short_raw"]).replace(0, np.nan)
    out["tilt"] = (out["p_long_raw"] - out["p_short_raw"]) / denom
    out["date"] = str(pd.Timestamp(last_date).date())
    return out.sort_values("p_long", ascending=False)


def pick_signals(scores, meta, n_side=5, min_tilt=0.0, margin=None):
    """Signals from calibrated success rates, gated per side."""
    thr_l = meta.get("threshold_long")
    thr_s = meta.get("threshold_short")
    G, L, c = meta["up"], meta["down"], 0.0014

    sig = []
    if thr_l is not None:
        longs = scores[(scores["p_long"] >= thr_l) &
                       (scores["tilt"] >= min_tilt)].nlargest(n_side, "p_long")
        for t, r in longs.iterrows():
            p = float(r["p_long"])
            sig.append({"ticker": t, "side": "long", "p": p,
                        "p_raw": round(float(r["p_long_raw"]), 4),
                        "threshold": round(float(thr_l), 4),
                        "tilt": round(float(r["tilt"]), 3),
                        "E_pct": round((p * G - (1 - p) * L - c) * 100, 3)})

    if thr_s is not None:
        held = {s["ticker"] for s in sig}
        shorts = scores[(scores["p_short"] >= thr_s) &
                        (scores["tilt"] <= -min_tilt)].nlargest(n_side, "p_short")
        shorts = shorts[~shorts.index.isin(held)]
        for t, r in shorts.iterrows():
            p = float(r["p_short"])
            sig.append({"ticker": t, "side": "short", "p": p,
                        "p_raw": round(float(r["p_short_raw"]), 4),
                        "threshold": round(float(thr_s), 4),
                        "tilt": round(float(r["tilt"]), 3),
                        "E_pct": round((p * G - (1 - p) * L - c) * 100, 3)})
    return sig


def load_models():
    import xgboost as xgb

    meta_path = MODEL_DIR / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError("No meta.json. Run the retrain-model workflow first.")
    meta = json.loads(meta_path.read_text())

    mp = MODEL_DIR / "model3.json"
    if not mp.exists():
        raise FileNotFoundError(
            "model3.json not found -- run the retrain-model workflow.")
    model = xgb.XGBClassifier()
    model.load_model(str(mp))
    return model, meta
