"""Out-of-sample evaluation for the 3-class barrier model.

THE DISTINCTION THAT MATTERS
----------------------------
A softmax output is a CLASS PROBABILITY. The quantity you trade on is a SUCCESS
RATE. They are related but not identical, and assuming they are equal is how a
model that looks tradeable turns out not to be.

    p_long   = softmax score for class LONG_WIN
    success  = P(actual class == LONG_WIN | we fired a long signal)

Regularisation, sample weighting and the multiclass normalisation all distort
the first relative to the second. So we MEASURE the mapping on out-of-sample
predictions -- separately for each side, because the long and short heads are
distorted differently:

    calibrated_P_long  = f_long(p_long)      fit against (y == LONG_WIN)
    calibrated_P_short = f_short(p_short)    fit against (y == SHORT_WIN)

Only the calibrated number may be compared against the breakeven threshold
P* = (L + c) / (G + L), because P* is a genuine probability.

WHAT PRECISION TO EXPECT
------------------------
Achievable precision is bounded jointly by AUC and base rate. With a ~33% base
rate:

    AUC 0.53  ->  top decile ~37%   (barely above base: near-random)
    AUC 0.64  ->  top decile ~55%,  top 2% ~63%
    AUC 0.70  ->  top decile ~64%,  top 2% ~76%

So a weak AUC caps selectivity no matter how the threshold is set, and a strong
AUC makes the top of the ranking genuinely sharp. Read the sweep with that in
mind: if the top bucket is not far above the base rate, the constraint is the
model's AUC, not the threshold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from qcs.cv import PurgedWalkForward
from qcs.labels3 import LONG_WIN, SHORT_WIN
from calibration import (calibration_curve, fit_isotonic, apply_calibration,
                         calibration_metrics, save_calibration)


def _auc(p, hit) -> float:
    df = pd.concat([pd.Series(np.asarray(p)).rename("p"),
                    pd.Series(np.asarray(hit)).rename("h")], axis=1).dropna()
    if df["h"].nunique() < 2:
        return float("nan")
    r = df["p"].rank()
    n1 = int((df["h"] == 1).sum()); n0 = int((df["h"] == 0).sum())
    return float((r[df["h"] == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def threshold_sweep(p, hit, p_star, G, L, c, n_years):
    """Precision / volume / expectancy at a ladder of thresholds."""
    p = np.asarray(p, dtype=float)
    hit = np.asarray(hit, dtype=float)
    ok = np.isfinite(p) & np.isfinite(hit)
    p, hit = p[ok], hit[ok]
    if len(p) < 500:
        return []

    rows = []
    for q in [0.50, 0.70, 0.80, 0.90, 0.95, 0.97, 0.98, 0.99, 0.995]:
        t = float(np.quantile(p, q))
        m = p >= t
        n = int(m.sum())
        if n < 30:
            continue
        prec = float(hit[m].mean())
        e = prec * G - (1 - prec) * L - c
        per_year = n / max(1e-9, n_years)
        rows.append({
            "quantile": q,
            "threshold": round(t, 4),
            "precision_pct": round(prec * 100, 2),
            "n_signals": n,
            "per_year": round(per_year, 1),
            "edge_vs_pstar_pp": round((prec - p_star) * 100, 2),
            "E_trade_pct": round(e * 100, 3),
        })
    return rows


def recommend(rows, min_signals=200, safety_pp=3.0):
    """Favour confidence over volume: the lowest threshold that clears P* by a
    safety margin AND rests on enough samples to be believable. A precision of
    90% measured on 11 signals is noise, not a finding."""
    ok = [r for r in rows if r["n_signals"] >= min_signals
          and r["edge_vs_pstar_pp"] >= safety_pp]
    if ok:
        return min(ok, key=lambda r: r["threshold"])
    enough = [r for r in rows if r["n_signals"] >= min_signals]
    if enough:
        b = max(enough, key=lambda r: r["precision_pct"])
        return b if b["edge_vs_pstar_pp"] > 0 else None
    return None


def evaluate3(X, y, cfg, fit_fn, p_star, G, L, n_splits=4):
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
    hit_l = (yy == LONG_WIN).astype(float)
    hit_s = (yy == SHORT_WIN).astype(float)

    # ---- per-side calibration: class probability -> observed success rate ----
    cal_l = fit_isotonic(P["p_long"], hit_l)
    cal_s = fit_isotonic(P["p_short"], hit_s)
    curve_l = calibration_curve(P["p_long"], hit_l)
    curve_s = calibration_curve(P["p_short"], hit_s)
    met_l = calibration_metrics(P["p_long"], hit_l)
    met_s = calibration_metrics(P["p_short"], hit_s)
    save_calibration(cal_l, cal_s, curve_l, curve_s, met_l, met_s)

    cL = pd.Series(apply_calibration(P["p_long"], cal_l), index=P.index)
    cS = pd.Series(apply_calibration(P["p_short"], cal_s), index=P.index)

    d = P.index.get_level_values("date")
    n_years = max(1.0, (d.max() - d.min()).days / 365.25)
    c = cfg.costs.round_trip

    sweep_l = threshold_sweep(cL, hit_l, p_star, G, L, c, n_years)
    sweep_s = threshold_sweep(cS, hit_s, p_star, G, L, c, n_years)
    rec_l, rec_s = recommend(sweep_l), recommend(sweep_s)

    def _by_year(cp, hit, rec):
        if not rec:
            return {}
        m = cp >= rec["threshold"]
        yrs = cp.index.get_level_values("date").year
        out = {}
        for v in sorted(set(yrs)):
            mm = m & (yrs == v)
            if mm.sum() >= 20:
                out[int(v)] = round(float(hit[mm].mean()) * 100, 1)
        return out

    return {
        "auc_long": round(_auc(P["p_long"], hit_l), 4),
        "auc_short": round(_auc(P["p_short"], hit_s), 4),
        "base_rate_long_pct": round(float(hit_l.mean()) * 100, 2),
        "base_rate_short_pct": round(float(hit_s.mean()) * 100, 2),
        "p_star_pct": round(p_star * 100, 2),
        "n_eval_samples": int(len(P)),
        "n_folds": len(preds),
        "n_years": round(n_years, 1),
        "calibration_long": met_l, "calibration_short": met_s,
        "calibration_curve_long": curve_l, "calibration_curve_short": curve_s,
        "sweep_long": sweep_l, "sweep_short": sweep_s,
        "recommended_long": rec_l, "recommended_short": rec_s,
        "threshold_long": rec_l["threshold"] if rec_l else None,
        "threshold_short": rec_s["threshold"] if rec_s else None,
        "precision_by_year_long": _by_year(cL, hit_l, rec_l),
        "mean_prob_sum": round(float(
            (P["p_long"] + P["p_short"] + P["p_neither"]).mean()), 6),
    }


def _print_side(name, sweep, rec, met, curve, p_star_pct, base_pct, auc):
    print(f"\n  --- {name} ---")
    print(f"  AUC {auc:.4f}   base rate {base_pct}%   breakeven P* {p_star_pct}%")
    if met:
        print(f"  calibration: ECE {met.get('ece')}  bias {met.get('bias'):+.4f} "
              f"({'under' if met.get('bias', 0) > 0 else 'over'}-confident)")
    if curve:
        print("  class probability -> observed success rate:")
        for r in curve[-5:]:
            print(f"    {r['predicted']:.3f} -> {r['observed']*100:>5.1f}%   "
                  f"lift {r['lift']:.2f}x   n={r['n']}")
    if not sweep:
        print("  (insufficient samples for a sweep)")
        return
    print(f"  {'q':>6} {'thresh':>7} {'precision':>10} {'edge':>8} "
          f"{'/year':>7} {'E/trade':>9}")
    for r in sweep:
        star = "  <=" if (rec and r["threshold"] == rec["threshold"]) else ""
        print(f"  {r['quantile']:>6.3f} {r['threshold']:>7.3f} "
              f"{r['precision_pct']:>9.1f}% {r['edge_vs_pstar_pp']:>+7.1f}p "
              f"{r['per_year']:>7.0f} {r['E_trade_pct']:>+8.2f}%{star}")
    if rec is None:
        print("  VERDICT: no threshold gives a reliable edge -> DO NOT TRADE this side.")
    else:
        print(f"  RECOMMENDED {rec['threshold']:.3f}: precision {rec['precision_pct']}%"
              f" (+{rec['edge_vs_pstar_pp']}pp), ~{rec['per_year']:.0f}/yr,"
              f" E={rec['E_trade_pct']:+.2f}%")


def print_report3(m, prev=None):
    if "error" in m:
        print(f"  evaluation failed: {m['error']}")
        return
    print("\n" + "=" * 74)
    print("OUT-OF-SAMPLE EVALUATION  (3-class, purged walk-forward)")
    print("=" * 74)
    print(f"  probability sum check: {m['mean_prob_sum']:.6f}  (must be 1.0)")

    _print_side("LONG", m["sweep_long"], m["recommended_long"],
                m["calibration_long"], m["calibration_curve_long"],
                m["p_star_pct"], m["base_rate_long_pct"], m["auc_long"])
    _print_side("SHORT", m["sweep_short"], m["recommended_short"],
                m["calibration_short"], m["calibration_curve_short"],
                m["p_star_pct"], m["base_rate_short_pct"], m["auc_short"])

    py = m.get("precision_by_year_long", {})
    if py:
        print("\n  LONG precision by year at the recommended threshold:")
        yrs = sorted(py)
        for i in range(0, len(yrs), 7):
            print("    " + "   ".join(f"{v}:{py[v]:.0f}%" for v in yrs[i:i+7]))
        below = sum(1 for v in py.values() if v < m["p_star_pct"])
        print(f"    -> {below} of {len(py)} years below breakeven")
