"""Model evaluation, run automatically on every retrain.

WHY A SEPARATE EVALUATION PASS
------------------------------
train.py deliberately fits the deployed model on ALL history -- for live
trading you want every scrap of data. But a model scored on its own training
data reports fantasy numbers. So we do two things:

    1. a purged walk-forward pass  -> honest OUT-OF-SAMPLE metrics
    2. a final fit on all data     -> the model that actually ships

The metrics come from (1); the model comes from (2). They are different objects
and conflating them is how people end up trusting a 0.95 AUC that was really
0.52.

THE METRIC THAT MATTERS: precision at the operating threshold
-------------------------------------------------------------
AUC tells you whether the ranking is any good. But you don't trade the ranking,
you trade the names that clear P* AND the tilt gate. So we measure precision at
exactly that operating point:

    expected_P_long = P(actual == LONG_WIN | model flagged it as a long signal)

That IS the expected success rate. Compare it to:
    p_star   -- below it, the signal loses money by construction
    base_rate -- below it, the model is worse than picking at random

And later compare it to the REALISED P from paper trading. Expected vs realised
is the single most informative diagnostic you can have: if expected P is 45%
and realised is 38%, the model's promise is not surviving contact with real
fills, and you know to look at slippage rather than at the model.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from qcs.cv import PurgedWalkForward
from qcs.labels3 import LONG_WIN, SHORT_WIN
from calibration import (calibration_curve, fit_isotonic, apply_calibration,
                         calibration_metrics, save_calibration)

HISTORY = Path(__file__).resolve().parent.parent / "model" / "training_history.json"


def _auc(p: pd.Series, hit: pd.Series) -> float:
    df = pd.concat([p.rename("p"), hit.rename("h")], axis=1).dropna()
    if df["h"].nunique() < 2:
        return float("nan")
    r = df["p"].rank()
    n1 = int((df["h"] == 1).sum())
    n0 = int((df["h"] == 0).sum())
    return float((r[df["h"] == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def evaluate(X, y, cfg, fit_fn, p_star, margin=0.02, min_tilt=0.10,
             n_splits=4):
    """Purged walk-forward evaluation. Returns a metrics dict."""
    dates = X.index.get_level_values("date").unique()
    splitter = PurgedWalkForward(n_splits, cfg.cv.purge,
                                 cfg.cv.embargo, cfg.cv.min_train)

    preds = []
    for tr_d, te_d in splitter.split(dates):
        tr = X.index.get_level_values("date").isin(tr_d)
        te = X.index.get_level_values("date").isin(te_d)
        Xtr, ytr = X[tr], y[tr]
        ok = ytr.notna() & (Xtr.notna().sum(axis=1) >= 0.5 * Xtr.shape[1])
        Xtr, ytr = Xtr[ok], ytr[ok].astype(int)
        if len(Xtr) < 300 or ytr.nunique() < 3:
            continue
        model = fit_fn(Xtr, ytr)
        Xte = X[te]
        Xte = Xte[Xte.notna().sum(axis=1) >= 0.5 * Xte.shape[1]]
        if len(Xte) == 0:
            continue
        P = model.predict_proba(Xte.to_numpy())
        preds.append(pd.DataFrame(
            {"p_short": P[:, SHORT_WIN], "p_neither": P[:, 1],
             "p_long": P[:, LONG_WIN]}, index=Xte.index))

    if not preds:
        return {"error": "no folds produced predictions"}

    P = pd.concat(preds).sort_index()
    yy = y.reindex(P.index)
    hit_long = (yy == LONG_WIN).astype(float)
    hit_short = (yy == SHORT_WIN).astype(float)

    # ---- CALIBRATION: softmax scores are NOT win rates until measured ----
    # Fit on these walk-forward (out-of-sample) predictions only.
    cal_l = fit_isotonic(P["p_long"], hit_long)
    cal_s = fit_isotonic(P["p_short"], hit_short)
    curve_l = calibration_curve(P["p_long"], hit_long)
    curve_s = calibration_curve(P["p_short"], hit_short)
    met_l = calibration_metrics(P["p_long"], hit_long)
    met_s = calibration_metrics(P["p_short"], hit_short)
    save_calibration(cal_l, cal_s, curve_l, curve_s, met_l, met_s)

    # CALIBRATED probabilities are what the trading gate must compare to P*.
    cP_long = pd.Series(apply_calibration(P["p_long"], cal_l), index=P.index)
    cP_short = pd.Series(apply_calibration(P["p_short"], cal_s), index=P.index)

    denom = (P["p_long"] + P["p_short"]).replace(0, np.nan)
    tilt = (P["p_long"] - P["p_short"]) / denom

    thr = p_star + margin

    # --- the operating point: both gates, exactly as pick_signals applies them
    flag_long = (cP_long > thr) & (tilt >= min_tilt)
    flag_short = (cP_short > thr) & (tilt <= -min_tilt)

    # for comparison: how many would the RAW (uncalibrated) gate have passed?
    raw_flag_long = (P["p_long"] > thr) & (tilt >= min_tilt)

    def _prec(flag, hit):
        n = int(flag.sum())
        if n == 0:
            return None, 0
        return float(hit[flag].mean()), n

    prec_l, n_l = _prec(flag_long, hit_long)
    prec_s, n_s = _prec(flag_short, hit_short)

    # per-year AUC, to watch decay across retrainings
    yr = P.index.get_level_values("date").year
    auc_year = {}
    for y_ in sorted(set(yr)):
        m = yr == y_
        if m.sum() < 50:
            continue
        a = _auc(P["p_long"][m], hit_long[m])
        if np.isfinite(a):
            auc_year[int(y_)] = round(a, 4)

    base_long = float(hit_long.mean())
    base_short = float(hit_short.mean())

    # expected edge per trade using the barrier geometry
    G, L = cfg_up(cfg), cfg_down(cfg)
    exp_e_long = (prec_l * G - (1 - prec_l) * L) if prec_l is not None else None

    return {
        "auc_long": round(_auc(P["p_long"], hit_long), 4),
        "auc_short": round(_auc(P["p_short"], hit_short), 4),
        "expected_P_long_pct": round(prec_l * 100, 2) if prec_l is not None else None,
        "expected_P_short_pct": round(prec_s * 100, 2) if prec_s is not None else None,
        "n_flagged_long": n_l,
        "n_flagged_short": n_s,
        "signal_rate_pct": round(100 * float((flag_long | flag_short).mean()), 2),
        "base_rate_long_pct": round(base_long * 100, 2),
        "base_rate_short_pct": round(base_short * 100, 2),
        "p_star_pct": round(p_star * 100, 2),
        "edge_vs_pstar_pp": round((prec_l - p_star) * 100, 2) if prec_l is not None else None,
        "expected_E_per_trade_pct": round(exp_e_long * 100, 3) if exp_e_long is not None else None,
        "auc_by_year": auc_year,
        "n_eval_samples": int(len(P)),
        "n_folds": len(preds),
        "calibration_long": met_l,
        "calibration_short": met_s,
        "calibration_curve_long": curve_l,
        "n_flagged_long_raw": int(raw_flag_long.sum()),
        "calibration_gain": (int(flag_long.sum()) - int(raw_flag_long.sum())),
    }


def cfg_up(cfg):
    return 0.06


def cfg_down(cfg):
    return 0.03


def append_history(entry: dict):
    """Append one retrain's metrics to the tracked history file."""
    hist = []
    if HISTORY.exists():
        try:
            hist = json.loads(HISTORY.read_text())
        except json.JSONDecodeError:
            hist = []
    hist.append(entry)
    hist = hist[-50:]                      # keep the last 50 retrainings
    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    HISTORY.write_text(json.dumps(hist, indent=2))
    return hist


def print_report(m: dict, prev: dict | None = None):
    """Human-readable summary, printed in the retrain workflow log."""
    if "error" in m:
        print(f"  evaluation failed: {m['error']}")
        return

    def delta(key, fmt="{:+.2f}"):
        if not prev or prev.get(key) is None or m.get(key) is None:
            return ""
        d = m[key] - prev[key]
        return f"   ({fmt.format(d)} vs last)"

    print("\n" + "=" * 62)
    print("OUT-OF-SAMPLE EVALUATION  (purged walk-forward)")
    print("=" * 62)
    print(f"  AUC long                 {m['auc_long']:.4f}{delta('auc_long', '{:+.4f}')}")
    print(f"  AUC short                {m['auc_short']:.4f}{delta('auc_short', '{:+.4f}')}")
    print()
    print("  --- EXPECTED SUCCESS RATE at the trading threshold ---")
    print(f"  expected P (long)        {m['expected_P_long_pct']}%"
          f"{delta('expected_P_long_pct')}")
    print(f"  expected P (short)       {m['expected_P_short_pct']}%"
          f"{delta('expected_P_short_pct')}")
    print(f"  breakeven P*             {m['p_star_pct']}%")
    print(f"  edge over breakeven      {m['edge_vs_pstar_pp']} pp"
          f"{delta('edge_vs_pstar_pp')}")
    print(f"  expected E per trade     {m['expected_E_per_trade_pct']}%")
    print()
    print(f"  base rate (long)         {m['base_rate_long_pct']}%   "
          f"<- beat this or the model adds nothing")

    cl = m.get("calibration_long") or {}
    if cl:
        print()
        print("  --- CALIBRATION (softmax score vs actual win rate) ---")
        print(f"  mean raw score           {cl.get('mean_pred')}")
        print(f"  actual base rate         {cl.get('base_rate')}")
        print(f"  bias                     {cl.get('bias'):+.4f}"
              f"   ({'UNDER' if cl.get('bias',0)>0 else 'OVER'}-confident)")
        print(f"  ECE                      {cl.get('ece')}   (0 = scores are true probabilities)")
        print(f"  Brier                    {cl.get('brier')}")
        if m.get("n_flagged_long_raw") is not None:
            print(f"  signals: raw gate {m['n_flagged_long_raw']} -> "
                  f"calibrated gate {m['n_flagged_long']}"
                  f"   ({m.get('calibration_gain', 0):+d})")
    cc = m.get("calibration_curve_long") or []
    if cc:
        print()
        print("  calibration curve (predicted -> observed win rate):")
        for r in cc:
            bar = "#" * int(round(r["observed"] * 40))
            print(f"    {r['predicted']:.3f} -> {r['observed']:.3f}  "
                  f"n={r['n']:>6}  lift={r['lift']:.2f}  {bar}")
    print(f"  signal rate              {m['signal_rate_pct']}% of name-days")
    print(f"  flagged long / short     {m['n_flagged_long']} / {m['n_flagged_short']}")

    if m.get("auc_by_year"):
        print("\n  AUC by year:")
        yrs = sorted(m["auc_by_year"])
        line = "   ".join(f"{y}:{m['auc_by_year'][y]:.3f}" for y in yrs[-8:])
        print(f"    {line}")

    # verdict
    print()
    if m["edge_vs_pstar_pp"] is None:
        print("  VERDICT: no signals cleared the threshold -- nothing to trade.")
    elif m["edge_vs_pstar_pp"] > 2:
        print(f"  VERDICT: expected P clears breakeven by "
              f"{m['edge_vs_pstar_pp']}pp. Tradeable on these numbers.")
    elif m["edge_vs_pstar_pp"] > 0:
        print(f"  VERDICT: thin margin ({m['edge_vs_pstar_pp']}pp over breakeven). "
              f"Real fills may erase it.")
    else:
        print(f"  VERDICT: expected P is BELOW breakeven. Do not trade this model.")
