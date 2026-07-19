"""Train per-side outcome models (WIN / STOP / TIMEOUT) with expectancy gating.

WHAT CHANGED AND WHY
--------------------
Previously the label answered "which side wins" -- {SHORT_WIN, NEITHER,
LONG_WIN}. That conflates two very different fates for a given trade: NEITHER is
about 60% stop-outs and 40% timeouts. Without separating them you cannot compute
expectancy, and the substitute test P* = (L+c)/(G+L) assumes every non-win is a
full -L loss, which is false whenever the time barrier binds -- and it made a
profitable model look unprofitable.

Now each side gets its own softmax over its OWN outcome space:

    {STOP, TIMEOUT, WIN}   ->   p_stop + p_timeout + p_win = 1

and the decision is economic rather than probabilistic:

    E = p_win*G - p_stop*L + p_timeout*r_timeout - c
    trade if E >= min_expectancy   (default 0.5%)

HORIZON = 10 DAYS
-----------------
At 5 days the time barrier bound hard: for a 2%-vol name, ~52% of paths reached
neither barrier, dragging the win rate to ~12%. Ten days roughly doubles the win
rate and cuts timeouts to ~24%, because a +/-6% move needs ~1.3 sigma at 5 days
but only ~0.95 sigma at 10.

Outputs:
    ../model/model_long.json, model_short.json
    ../model/meta.json, calibration.json, training_history.json
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
from qcs.labels_side import side_outcome_labels, timeout_returns, outcome_distribution
from evaluate_side import evaluate_side, print_report_side
from evaluate import append_history, HISTORY
from backtest_sim import simulate_history

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
MODEL_DIR.mkdir(exist_ok=True)

UP, DOWN = 0.06, 0.03
VOL_CAP_COLS = {"rvol_252", "rvol_60", "rvol_ratio", "rvol_ratio_long",
                "vol_of_vol", "mkt_stress", "mkt_stress_z"}


def make_fitter(cfg, feature_cols):
    m = cfg.model
    fw = np.ones(len(feature_cols))
    for i, c in enumerate(feature_cols):
        if c in VOL_CAP_COLS:
            fw[i] = cfg.model_vol_cap_weight

    def _fit(Xtr, ytr):
        mdl = xgb.XGBClassifier(
            n_estimators=m.n_estimators, max_depth=m.max_depth,
            learning_rate=m.learning_rate, subsample=m.subsample,
            colsample_bytree=m.colsample_bytree,
            min_child_weight=m.min_child_weight,
            reg_lambda=m.reg_lambda, random_state=m.random_state,
            objective="multi:softprob", num_class=3, eval_metric="mlogloss",
            feature_weights=fw, n_jobs=4, verbosity=0,
        )
        # only the label-overlap correction; no class weighting (it would
        # decalibrate the probabilities that expectancy depends on)
        mdl.fit(Xtr.to_numpy(), ytr.to_numpy().astype(int),
                sample_weight=np.full(len(ytr), 1.0 / cfg.label.horizon))
        return mdl

    return _fit


def main():
    cfg = Config()
    H = cfg.label.horizon
    print(f"Downloading {len(cfg.universe)} names...  horizon={H}d  "
          f"barriers +{UP:.0%}/-{DOWN:.0%}")
    px = load_yahoo(cfg.universe, cfg.start, cfg.end)
    fx = load_yahoo(cfg.sector_etfs, cfg.start, cfg.end)
    sx = load_yahoo(cfg.stress_tickers, cfg.start, cfg.end)

    close, volume = px["close"], px["volume"]
    ok = close.notna().sum() > 400
    close, volume = close.loc[:, ok], volume.loc[:, ok]
    print(f"  {close.shape[1]} names x {close.shape[0]} days")

    print("Building features...")
    X = cross_sectional_zscore(
        build_features(close, volume, fx["close"], stress_close=sx["close"]))

    results, models, dists = {}, {}, {}
    for side in ("long", "short"):
        y = side_outcome_labels(close, UP, DOWN, H, side)
        rt = timeout_returns(close, UP, DOWN, H, side)

        idx = X.index.intersection(y.index)
        Xs, ys = X.loc[idx].sort_index(), y.loc[idx].sort_index()
        mask = ys.notna() & (Xs.notna().sum(axis=1) >= 0.5 * Xs.shape[1])
        Xf, yf = Xs[mask], ys[mask].astype(int)
        dists[side] = outcome_distribution(yf)
        print(f"\n{side.upper()}: {len(Xf)} samples  {dists[side]}")

        fit = make_fitter(cfg, list(Xf.columns))
        print(f"  evaluating out-of-sample...")
        m = evaluate_side(Xf, yf, rt.reindex(Xf.index), cfg, fit, UP, DOWN, side)
        print_report_side(m)
        results[side] = m

        print(f"  fitting final {side} model on all data...")
        models[side] = fit(Xf, yf)
        models[side].save_model(str(MODEL_DIR / f"model_{side}.json"))
        feat_cols = list(Xf.columns)

    # calibration maps for live use
    (MODEL_DIR / "calibration.json").write_text(json.dumps({
        "long": results["long"].get("cal_win", {}),
        "long_stop": results["long"].get("cal_stop", {}),
        "short": results["short"].get("cal_win", {}),
        "short_stop": results["short"].get("cal_stop", {}),
        "curve_long": results["long"].get("calibration_curve_win", []),
        "curve_short": results["short"].get("calibration_curve_win", []),
        "metrics_long": results["long"].get("calibration_win", {}),
        "metrics_short": results["short"].get("calibration_win", {}),
    }, indent=2))

    meta = {
        "model_type": "per_side_outcome",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "features": feat_cols,
        "universe": list(close.columns),
        "up": UP, "down": DOWN, "horizon": H,
        "min_expectancy": cfg.min_expectancy,
        "min_E_long": results["long"].get("min_E"),
        "min_E_short": results["short"].get("min_E"),
        "r_timeout_long": results["long"].get("r_timeout_pct", 0.0) / 100.0,
        "r_timeout_short": results["short"].get("r_timeout_pct", 0.0) / 100.0,
        "cost": cfg.costs.round_trip,
        "outcome_distribution": dists,
        "train_start": str(close.index[0].date()),
        "train_end": str(close.index[-1].date()),
        "n_samples": int(len(Xf)),
    }
    (MODEL_DIR / "meta.json").write_text(json.dumps(meta, indent=2))

    # ---- historical portfolio simulation on the OOS predictions ----
    print("\nRunning historical portfolio simulation (out-of-sample)...")
    preds = {s: results[s]["oos_predictions"] for s in ("long", "short")
             if "oos_predictions" in results[s]}
    bt = simulate_history(preds, close, meta) if preds else None
    if bt:
        st = bt["stats"]
        print(f"  {st['period_start']} -> {st['period_end']}  ({st['years']}y)")
        print(f"  final equity  ${st['final_equity']:,.0f}  "
              f"({st['total_return_pct']:+.1f}%, CAGR {st['cagr_pct']:+.1f}%)")
        print(f"  Sharpe {st['sharpe']}  maxDD {st['max_drawdown_pct']}%  "
              f"trades {st['n_trades']} ({st['trades_per_year']}/yr)")
        print(f"  win rate {st['win_rate_pct']}%  avg win {st['avg_win_pct']}%  "
              f"avg loss {st['avg_loss_pct']}%  PF {st['profit_factor']}")
        bt_out = {"generated_at": meta["trained_at"], **bt,
                  "trades": bt["trades"][-500:]}
        (MODEL_DIR / "backtest.json").write_text(json.dumps(bt_out, indent=2))

    strip = lambda m: {k: v for k, v in m.items()
                       if k not in ("cal_win", "cal_stop", "calibration_curve_win",
                                    "oos_predictions")}
    append_history({
        "trained_at": meta["trained_at"], "train_end": meta["train_end"],
        "n_samples": meta["n_samples"], "n_names": len(meta["universe"]),
        "n_features": len(feat_cols), "horizon": H,
        "metrics": {"long": strip(results["long"]), "short": strip(results["short"])},
    })

    for old in ("model3.json",):
        p = MODEL_DIR / old
        if p.exists():
            p.unlink(); print(f"  removed stale {old}")

    print(f"\nSaved. gates: long E>={meta['min_E_long']}  short E>={meta['min_E_short']}")


if __name__ == "__main__":
    main()
