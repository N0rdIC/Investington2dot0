"""Probability calibration.

THE PROBLEM
-----------
A softmax output is a CLASS SCORE, not a success rate. XGBoost with strong
regularisation (shallow trees, high min_child_weight, L2 penalty) systematically
shrinks predictions toward the base rate, so a raw p_long of 0.40 may correspond
to an actual win rate of 0.46, or 0.34 -- the mapping has to be MEASURED, never
assumed.

This matters far beyond reporting. The trading gate is

    enter long if P(win) > P* = (L + c) / (G + L)

If we feed a shrunk class score into that comparison, we are testing the wrong
quantity. An under-confident model will almost never clear the threshold and the
strategy will appear to have no opportunities -- which is exactly the symptom we
saw (82% of rebalance periods produced zero trades).

THE FIX
-------
Fit an isotonic regression on OUT-OF-SAMPLE predictions:

    calibrated_P = f(raw_p)     where f is monotone non-decreasing

Isotonic is the right tool here: it makes no shape assumption beyond monotonicity
(a higher score should never mean a lower win rate), and with tens of thousands
of samples it has plenty of data to fit a flexible mapping.

The calibrator is fit on walk-forward predictions ONLY. Fitting it on in-sample
predictions would produce a mapping that looks perfect and lies in production.

It is stored as interpolation breakpoints rather than a pickled sklearn object,
so it survives library upgrades and is human-readable in git.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

CALIB_PATH = Path(__file__).resolve().parent.parent / "model" / "calibration.json"


def calibration_curve(p, hit, n_bins: int = 10) -> list[dict]:
    """Equal-frequency bins: mean predicted score vs observed win rate.

    This is the diagnostic the user reads. If `predicted` and `observed` track
    each other, the model is calibrated. If observed is consistently higher, the
    model is under-confident and the raw threshold test is too strict.
    """
    p = np.asarray(p, dtype=float)
    hit = np.asarray(hit, dtype=float)
    ok = np.isfinite(p) & np.isfinite(hit)
    p, hit = p[ok], hit[ok]
    if len(p) < n_bins * 10:
        return []

    qs = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    qs[0] -= 1e-9
    qs[-1] += 1e-9
    idx = np.digitize(p, qs[1:-1], right=True)

    rows = []
    for b in range(n_bins):
        m = idx == b
        n = int(m.sum())
        if n < 20:
            continue
        rows.append({
            "bin": b,
            "n": n,
            "p_lo": round(float(p[m].min()), 4),
            "p_hi": round(float(p[m].max()), 4),
            "predicted": round(float(p[m].mean()), 4),
            "observed": round(float(hit[m].mean()), 4),
            "lift": round(float(hit[m].mean() / max(1e-9, hit.mean())), 3),
        })
    return rows


def fit_isotonic(p, hit, n_points: int = 40) -> dict:
    """Fit a monotone mapping raw score -> win rate. Returns breakpoints."""
    from sklearn.isotonic import IsotonicRegression

    p = np.asarray(p, dtype=float)
    hit = np.asarray(hit, dtype=float)
    ok = np.isfinite(p) & np.isfinite(hit)
    p, hit = p[ok], hit[ok]
    if len(p) < 200:
        return {}

    ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    ir.fit(p, hit)

    lo, hi = float(np.quantile(p, 0.001)), float(np.quantile(p, 0.999))
    xs = np.linspace(lo, hi, n_points)
    ys = ir.predict(xs)
    return {"x": [round(float(v), 5) for v in xs],
            "y": [round(float(v), 5) for v in ys]}


def apply_calibration(p, calib: dict):
    """Map raw scores to calibrated win rates via linear interpolation."""
    if not calib or "x" not in calib or not calib["x"]:
        return np.asarray(p, dtype=float)
    return np.interp(np.asarray(p, dtype=float), calib["x"], calib["y"])


def calibration_metrics(p, hit) -> dict:
    """Brier score and Expected Calibration Error.

    Brier = mean squared error of the probability (lower is better).
    ECE    = average |predicted - observed| across bins, weighted by bin size.
             ECE near 0 means the scores can be read as probabilities directly.
    """
    p = np.asarray(p, dtype=float)
    hit = np.asarray(hit, dtype=float)
    ok = np.isfinite(p) & np.isfinite(hit)
    p, hit = p[ok], hit[ok]
    if len(p) == 0:
        return {}

    brier = float(np.mean((p - hit) ** 2))
    rows = calibration_curve(p, hit, n_bins=10)
    if rows:
        tot = sum(r["n"] for r in rows)
        ece = sum(r["n"] * abs(r["predicted"] - r["observed"]) for r in rows) / tot
        # average signed gap: positive => model is UNDER-confident
        bias = sum(r["n"] * (r["observed"] - r["predicted"]) for r in rows) / tot
    else:
        ece = bias = float("nan")

    return {
        "brier": round(brier, 5),
        "ece": round(float(ece), 5),
        "bias": round(float(bias), 5),
        "base_rate": round(float(hit.mean()), 5),
        "mean_pred": round(float(p.mean()), 5),
    }


def save_calibration(calib_long: dict, calib_short: dict,
                     curve_long: list, curve_short: list,
                     metrics_long: dict, metrics_short: dict):
    CALIB_PATH.parent.mkdir(parents=True, exist_ok=True)
    CALIB_PATH.write_text(json.dumps({
        "long": calib_long, "short": calib_short,
        "curve_long": curve_long, "curve_short": curve_short,
        "metrics_long": metrics_long, "metrics_short": metrics_short,
    }, indent=2))


def load_calibration() -> dict:
    if not CALIB_PATH.exists():
        return {}
    try:
        return json.loads(CALIB_PATH.read_text())
    except json.JSONDecodeError:
        return {}
