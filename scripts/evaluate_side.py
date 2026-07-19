"""Out-of-sample evaluation with EXPECTANCY-based gating.

The decision quantity is no longer a probability compared to a threshold. It is
the expected value of the trade itself:

    E = p_win*G - p_stop*L + p_timeout*r_timeout - c

all three probabilities coming from one per-side softmax over
{STOP, TIMEOUT, WIN}. We then sweep the MINIMUM E required to trade and report,
for each level, how many signals survive and what they actually earned.

Two things this fixes relative to the previous version:

  1. p_stop is now predicted rather than assumed. Previously every non-win was
     treated as a full -L loss, which is wrong whenever trades time out.
  2. The reported "realised E" is measured from actual barrier outcomes, so the
     sweep shows whether the model's predicted E is trustworthy -- predicted vs
     realised E is the honest scorecard.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from qcs.cv import PurgedWalkForward
from qcs.labels_side import STOP, TIMEOUT, WIN, true_breakeven
from calibration import (calibration_curve, fit_isotonic, apply_calibration,
                         calibration_metrics)


def _auc(p, hit) -> float:
    df = pd.concat([pd.Series(np.asarray(p)).rename("p"),
                    pd.Series(np.asarray(hit)).rename("h")], axis=1).dropna()
    if df["h"].nunique() < 2:
        return float("nan")
    r = df["p"].rank()
    n1 = int((df["h"] == 1).sum()); n0 = int((df["h"] == 0).sum())
    return float((r[df["h"] == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def walk_forward(X, y, cfg, fit_fn, n_splits=4):
    dates = X.index.get_level_values("date").unique()
    sp = PurgedWalkForward(n_splits, cfg.cv.purge, cfg.cv.embargo, cfg.cv.min_train)
    preds = []
    for tr_d, te_d in sp.split(dates):
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
            {"p_stop": P[:, STOP], "p_timeout": P[:, TIMEOUT], "p_win": P[:, WIN]},
            index=Xte.index))
    if not preds:
        return None
    return pd.concat(preds).sort_index()


def evaluate_side(X, y, r_timeout, cfg, fit_fn, G, L, side="long", n_splits=4):
    P = walk_forward(X, y, cfg, fit_fn, n_splits)
    if P is None:
        return {"error": "no folds produced predictions"}

    yy = y.reindex(P.index)
    hit_win = (yy == WIN).astype(float)
    hit_stop = (yy == STOP).astype(float)
    c = cfg.costs.round_trip

    # --- calibrate the two probabilities that carry economic weight ---
    cal_w = fit_isotonic(P["p_win"], hit_win)
    cal_s = fit_isotonic(P["p_stop"], hit_stop)
    cw = pd.Series(apply_calibration(P["p_win"], cal_w), index=P.index)
    cs = pd.Series(apply_calibration(P["p_stop"], cal_s), index=P.index)

    # measured average return of trades that time out (usually small, not zero)
    rt = r_timeout.reindex(P.index)
    r_time_mean = float(rt.dropna().mean()) if rt.notna().any() else 0.0

    # predicted expectancy per name-day
    p_time = (1.0 - cw - cs).clip(lower=0.0)
    E_pred = cw * G - cs * L + p_time * r_time_mean - c

    # realised outcome per name-day, using the barrier payoffs
    realised = np.where(yy == WIN, G,
                np.where(yy == STOP, -L, rt.fillna(0.0))) - c
    realised = pd.Series(realised, index=P.index)

    base = {
        "win_pct": round(float(hit_win.mean()) * 100, 2),
        "stop_pct": round(float(hit_stop.mean()) * 100, 2),
        "timeout_pct": round(float((yy == TIMEOUT).mean()) * 100, 2),
    }
    naive_pstar = (L + c) / (G + L)
    true_be = true_breakeven(float(hit_stop.mean()), G, L, c)

    d = P.index.get_level_values("date")
    n_years = max(1.0, (d.max() - d.min()).days / 365.25)

    # --- sweep the minimum-E gate ---
    rows = []
    for min_e in [0.000, 0.0025, 0.005, 0.0075, 0.010, 0.015, 0.020, 0.030]:
        m = E_pred >= min_e
        n = int(m.sum())
        if n < 30:
            continue
        rows.append({
            "min_E_pct": round(min_e * 100, 2),
            "n_signals": n,
            "per_year": round(n / n_years, 1),
            "win_rate_pct": round(float(hit_win[m].mean()) * 100, 2),
            "stop_rate_pct": round(float(hit_stop[m].mean()) * 100, 2),
            "E_predicted_pct": round(float(E_pred[m].mean()) * 100, 3),
            "E_realised_pct": round(float(realised[m].mean()) * 100, 3),
        })

    # ---- pick the operating gate -------------------------------------------
    # HARD FLOOR: never recommend a gate below the configured min_expectancy.
    # The earlier version fell back to the lowest gate with positive realised E
    # whenever nothing cleared the floor, which silently discarded the user's
    # risk preference and produced a "recommendation" of E >= 0 -- i.e. trade
    # everything. If nothing at or above the floor earns money out-of-sample,
    # the correct answer is to trade nothing.
    floor = getattr(cfg, "min_expectancy", 0.005)
    eligible = [r for r in rows
                if r["min_E_pct"] >= floor * 100 - 1e-9      # respect the floor
                and r["n_signals"] >= 150                    # enough to believe
                and r["E_realised_pct"] > 0]                 # actually made money
    rec = min(eligible, key=lambda r: r["min_E_pct"]) if eligible else None

    # diagnostic: does higher conviction actually help? If realised E falls as
    # the gate rises, the ranking is anti-predictive and no gate will save it.
    monotone = None
    if len(rows) >= 2:
        er = [r["E_realised_pct"] for r in rows]
        monotone = bool(er[-1] >= er[0])

    return {
        "side": side,
        "oos_predictions": oos,
        "auc_win": round(_auc(P["p_win"], hit_win), 4),
        "auc_stop": round(_auc(P["p_stop"], hit_stop), 4),
        "base": base,
        "naive_pstar_pct": round(naive_pstar * 100, 2),
        "true_breakeven_pct": round(true_be * 100, 2),
        "r_timeout_pct": round(r_time_mean * 100, 3),
        "calibration_win": calibration_metrics(P["p_win"], hit_win),
        "calibration_curve_win": calibration_curve(P["p_win"], hit_win),
        "cal_win": cal_w, "cal_stop": cal_s,
        "sweep": rows,
        "recommended": rec,
        "min_E": rec["min_E_pct"] / 100 if rec else None,
        "floor_pct": round(getattr(cfg, "min_expectancy", 0.005) * 100, 2),
        "gate_monotone": monotone,
        "n_eval": int(len(P)),
        "n_years": round(n_years, 1),
    }


def print_report_side(m):
    if "error" in m:
        print(f"  {m['side']}: evaluation failed: {m['error']}")
        return
    b = m["base"]
    print(f"\n  ===== {m['side'].upper()} =====")
    print(f"  outcomes: WIN {b['win_pct']}%  STOP {b['stop_pct']}%  "
          f"TIMEOUT {b['timeout_pct']}%")
    print(f"  naive P* (wrong: assumes no timeouts)  {m['naive_pstar_pct']}%")
    print(f"  TRUE breakeven given the stop rate     {m['true_breakeven_pct']}%")
    print(f"  mean return when timing out            {m['r_timeout_pct']}%")
    print(f"  AUC  win {m['auc_win']:.4f}   stop {m['auc_stop']:.4f}")
    cw = m.get("calibration_win") or {}
    if cw:
        print(f"  calibration(win): ECE {cw.get('ece')}  bias {cw.get('bias'):+.4f}")

    print(f"\n  {'min E':>7} {'signals':>8} {'/yr':>6} {'win%':>7} {'stop%':>7} "
          f"{'E pred':>8} {'E real':>8}")
    for r in m["sweep"]:
        star = "  <=" if (m.get("recommended") and
                          r["min_E_pct"] == m["recommended"]["min_E_pct"]) else ""
        print(f"  {r['min_E_pct']:>6.2f}% {r['n_signals']:>8} {r['per_year']:>6.0f} "
              f"{r['win_rate_pct']:>6.1f}% {r['stop_rate_pct']:>6.1f}% "
              f"{r['E_predicted_pct']:>+7.2f}% {r['E_realised_pct']:>+7.2f}%{star}")

    if m.get("gate_monotone") is False:
        print("  WARNING: realised E FALLS as the gate rises -- the expectancy")
        print("           ranking is anti-predictive. Raising conviction makes it worse.")
    rec = m.get("recommended")
    if rec is None:
        print(f"  VERDICT: no gate at or above the {m.get('floor_pct')}% floor has a")
        print(f"           positive realised E -> DO NOT TRADE this side.")
    else:
        print(f"  RECOMMENDED gate E >= {rec['min_E_pct']:.2f}%  ->  "
              f"realised E {rec['E_realised_pct']:+.2f}%/trade, "
              f"~{rec['per_year']:.0f} signals/yr, win rate {rec['win_rate_pct']:.1f}%")
