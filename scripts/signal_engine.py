"""Signal engine: per-side outcome models -> expectancy -> signals.

The decision is economic, not probabilistic. For each name and each side:

    E = p_win*G - p_stop*L + p_timeout*r_timeout - c

using calibrated p_win and p_stop from that side's softmax over
{STOP, TIMEOUT, WIN}. A signal fires only when E clears `min_expectancy`
(default 0.5%), which leaves room for estimation error, gap slippage on stops,
and borrow cost on shorts.

Note this replaces the old test `p > P* = (L+c)/(G+L)`. That formula assumes
every non-win is a full -L loss, which is false whenever trades time out -- with
a 10-day horizon roughly a quarter of them do.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from qcs.features import build_features, cross_sectional_zscore
from calibration import load_calibration, apply_calibration

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
STOP, TIMEOUT, WIN = 0, 1, 2


def compute_features(close, volume, factor_close, stress_close=None):
    panel = build_features(close, volume, factor_close, stress_close=stress_close)
    return cross_sectional_zscore(panel)


def realized_sigma(close, lookback=126):
    """Daily volatility over ~6 months, per name.

    Reported alongside every signal because barrier reachability depends on it:
    with +/-6% barriers over 10 days, sigma_d=2.0% makes the target a 0.95-sigma
    move (routinely hit), while sigma_d=1.2% makes it 1.6-sigma (rarely hit, so
    the trade just times out and pays commission for nothing).
    """
    r = close.pct_change()
    return r.tail(lookback).std()


def latest_scores(close, volume, factor_close, stress_close, models, meta):
    """Calibrated probabilities and expectancy per ticker, for both sides."""
    X = compute_features(close, volume, factor_close, stress_close)
    last_date = X.index.get_level_values("date").max()
    Xt = X.xs(last_date, level="date").reindex(columns=meta["features"])
    Xt = Xt[Xt.notna().sum(axis=1) >= 0.5 * len(meta["features"])]

    cal = load_calibration()
    G, L, c = meta["up"], meta["down"], meta.get("cost", 0.0014)
    out = pd.DataFrame(index=Xt.index)

    # 6-month realised vol, and how many sigma the barrier represents
    sig = realized_sigma(close).reindex(Xt.index)
    H = meta.get("horizon", 10)
    out["sigma_d"] = sig
    out["sigma_h"] = sig * np.sqrt(H)
    out["barrier_sigmas"] = G / out["sigma_h"].replace(0, np.nan)

    for side in ("long", "short"):
        P = models[side].predict_proba(Xt.to_numpy())
        pw = apply_calibration(P[:, WIN], cal.get(side, {}))
        ps = apply_calibration(P[:, STOP], cal.get(f"{side}_stop", {}))
        pt = np.clip(1.0 - pw - ps, 0.0, 1.0)
        rt = meta.get(f"r_timeout_{side}", 0.0)

        out[f"p_win_{side}"] = pw
        out[f"p_stop_{side}"] = ps
        out[f"p_time_{side}"] = pt
        out[f"E_{side}"] = pw * G - ps * L + pt * rt - c

    out["date"] = str(pd.Timestamp(last_date).date())
    return out.sort_values("E_long", ascending=False)


def pick_signals(scores, meta, n_side=5, min_tilt=None, margin=None):
    """Fire only where predicted expectancy clears the validated gate."""
    G, L = meta["up"], meta["down"]
    default_gate = meta.get("min_expectancy", 0.005)
    min_sig = meta.get("min_sigma_daily", 0.0) or 0.0
    sig = []

    # volatility floor: a name that cannot plausibly reach the barrier within
    # the horizon will just time out, paying commission for no outcome.
    if min_sig > 0 and "sigma_d" in scores.columns:
        scores = scores[scores["sigma_d"].fillna(0) >= min_sig]

    for side in ("long", "short"):
        gate = meta.get(f"min_E_{side}")
        if gate is None:
            continue                       # side not validated -> trade nothing
        gate = max(gate, default_gate)
        col = f"E_{side}"
        held = {s["ticker"] for s in sig}
        cand = scores[scores[col] >= gate].nlargest(n_side, col)
        cand = cand[~cand.index.isin(held)]
        for t, r in cand.iterrows():
            sig.append({
                "ticker": t, "side": side,
                "sigma_d_pct": round(float(r.get("sigma_d", float("nan"))) * 100, 2)
                                if pd.notna(r.get("sigma_d")) else None,
                "barrier_sigmas": round(float(r.get("barrier_sigmas", float("nan"))), 2)
                                if pd.notna(r.get("barrier_sigmas")) else None,
                "p": round(float(r[f"p_win_{side}"]), 4),
                "p_stop": round(float(r[f"p_stop_{side}"]), 4),
                "p_time": round(float(r[f"p_time_{side}"]), 4),
                "E_pct": round(float(r[col]) * 100, 3),
                "gate_pct": round(gate * 100, 2),
            })
    return sig


def load_models():
    import xgboost as xgb

    mp = MODEL_DIR / "meta.json"
    if not mp.exists():
        raise FileNotFoundError("No meta.json. Run the retrain-model workflow first.")
    meta = json.loads(mp.read_text())

    models = {}
    for side in ("long", "short"):
        f = MODEL_DIR / f"model_{side}.json"
        if not f.exists():
            raise FileNotFoundError(
                f"model_{side}.json not found -- run the retrain-model workflow.")
        m = xgb.XGBClassifier()
        m.load_model(str(f))
        models[side] = m
    return models, meta
