"""Barrier-event backtest (classification).

Distinct from backtest.py (which is the cross-sectional RETURN ranker). Here:

  - target is the 0/1 triple-barrier label (does +G hit before -L in `horizon`?)
  - model is a CLASSIFIER; its output is the predicted probability P(event)
  - we run TWO models: one for the long barrier, one for the short barrier
  - a name enters the long book when P_long is high AND clears the breakeven P*
    implied by the G/L geometry; the short book mirrors it
  - position P&L uses the SAME barrier logic to realise G or -L per trade

This is the design that matches the original P*G - (1-P)*L - c framework
literally: the model estimates P, the barriers define G and L.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from .config import Config
from .cv import PurgedWalkForward
from .features import build_features, cross_sectional_zscore
from .labels import triple_barrier_labels, barrier_base_rate


def _fit_clf(X, y, cfg, scale_pos_weight=1.0):
    m = cfg.model

    # VOL-FEATURE CAP. Rather than dropping the volatility features (which carry
    # real information), we down-weight their probability of being chosen at
    # each split via XGBoost's feature_weights. The vol-level features can still
    # be used, but they can no longer dominate every tree — forcing the model to
    # find directional signal in the other features too. This is the "cap so
    # they can't dominate" choice, implemented at the column-sampling level.
    capped = {
        "rvol_252", "rvol_60", "rvol_ratio", "rvol_ratio_long", "rvol_252",
        "vol_of_vol", "mkt_stress", "mkt_stress_z",
    }
    fw = np.ones(X.shape[1], dtype=float)
    for i, col in enumerate(X.columns):
        if col in capped:
            fw[i] = cfg.model_vol_cap_weight  # e.g. 0.25 -> quarter the odds
    model = xgb.XGBClassifier(
        n_estimators=m.n_estimators,
        max_depth=m.max_depth,
        learning_rate=m.learning_rate,
        subsample=m.subsample,
        colsample_bytree=m.colsample_bytree,
        min_child_weight=m.min_child_weight,
        reg_lambda=m.reg_lambda,
        random_state=m.random_state,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        feature_weights=fw,
        n_jobs=4,
        verbosity=0,
    )
    w = np.full(len(y), 1.0 / cfg.label.horizon)
    model.fit(X, y, sample_weight=w)
    return model


def walk_forward_proba(X, y, cfg):
    """Out-of-sample predicted P(event) via purged walk-forward."""
    dates = X.index.get_level_values("date").unique()
    splitter = PurgedWalkForward(cfg.cv.n_splits, cfg.cv.purge,
                                 cfg.cv.embargo, cfg.cv.min_train)
    preds, imps = [], []
    for tr_d, te_d in splitter.split(dates):
        tr = X.index.get_level_values("date").isin(tr_d)
        te = X.index.get_level_values("date").isin(te_d)
        Xtr, ytr = X[tr], y[tr]
        ok = ytr.notna() & (Xtr.notna().sum(axis=1) >= 0.5 * Xtr.shape[1])
        Xtr, ytr = Xtr[ok], ytr[ok].astype(int)
        if len(Xtr) < 300 or ytr.nunique() < 2:
            continue
        # class imbalance: events are the minority, weight them up
        spw = float((ytr == 0).sum()) / max(1, (ytr == 1).sum())
        model = _fit_clf(Xtr, ytr, cfg, scale_pos_weight=spw)
        Xte = X[te]
        Xte = Xte[Xte.notna().sum(axis=1) >= 0.5 * Xte.shape[1]]
        if len(Xte) == 0:
            continue
        p = pd.Series(model.predict_proba(Xte)[:, 1], index=Xte.index)
        preds.append(p)
        imps.append(pd.Series(model.feature_importances_, index=X.columns))
    if not preds:
        raise RuntimeError("no folds produced predictions")
    return pd.concat(preds).sort_index(), pd.concat(imps, axis=1).mean(axis=1).sort_values(ascending=False)


def realised_barrier_return(close, dt, ticker, up, down, horizon, side):
    """The actual G or -L this trade earns, using the barrier path.

    Returns the signed return (before costs). For a long: +up if the up barrier
    is hit first, -down if the stop is hit first, else the raw h-day return
    (time barrier: exit at horizon at whatever price prevails)."""
    try:
        p = close[ticker]
    except KeyError:
        return np.nan
    if dt not in p.index:
        return np.nan
    i = p.index.get_loc(dt)
    p0 = p.iloc[i]
    if not np.isfinite(p0) or p0 <= 0:
        return np.nan
    end = min(i + horizon, len(p) - 1)
    for s in range(i + 1, end + 1):
        ps = p.iloc[s]
        if not np.isfinite(ps):
            continue
        r = ps / p0 - 1.0
        if side == "long":
            if r >= up:
                return up
            if r <= -down:
                return -down
        else:
            if r <= -up:
                return up            # profitable short returns +up
            if r >= down:
                return -down
    # time barrier: raw return at horizon (signed for side)
    ps = p.iloc[end]
    r = ps / p0 - 1.0 if np.isfinite(ps) else 0.0
    return r if side == "long" else -r


def run_barrier(close, volume, factor_close, cfg,
                up=0.06, down=0.03, shuffle=False, stress_close=None):
    """Full barrier backtest for both books."""
    X = cross_sectional_zscore(
        build_features(close, volume, factor_close, stress_close=stress_close))

    y_long = triple_barrier_labels(close, up, down, cfg.label.horizon, "long")
    y_short = triple_barrier_labels(close, up, down, cfg.label.horizon, "short")

    idx = X.index.intersection(y_long.index)
    X = X.loc[idx].sort_index()
    y_long = y_long.loc[idx].sort_index()
    y_short = y_short.loc[idx].sort_index()

    if shuffle:
        rng = np.random.default_rng(cfg.model.random_state)
        y_long = y_long.groupby(level="date").transform(
            lambda s: pd.Series(rng.permutation(s.to_numpy()), index=s.index))
        y_short = y_short.groupby(level="date").transform(
            lambda s: pd.Series(rng.permutation(s.to_numpy()), index=s.index))

    base_long = barrier_base_rate(y_long)
    base_short = barrier_base_rate(y_short)

    p_long, imp_long = walk_forward_proba(X, y_long, cfg)
    p_short, imp_short = walk_forward_proba(X, y_short, cfg)

    # breakeven P* from geometry
    c = cfg.costs.round_trip
    p_star = (down + c) / (up + down)

    # --- AUC: does predicted P rank actual events? (the honest signal test) ---
    def _auc(p, y):
        df = pd.concat([p.rename("p"), y.rename("y")], axis=1).dropna()
        df = df[df["y"].isin([0, 1])]
        if df["y"].nunique() < 2:
            return np.nan
        # rank-based AUC
        r = df["p"].rank()
        n1 = (df["y"] == 1).sum(); n0 = (df["y"] == 0).sum()
        return float((r[df["y"] == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

    auc_long = _auc(p_long, y_long)
    auc_short = _auc(p_short, y_short)

    # --- PER-YEAR AUC: is the skill stable across regimes, or all in the
    # 2020-21 melt-up? A signal that is >0.6 every year (incl. 2022) is real;
    # one that lives only in the bull years is a regime bet. ---
    def _auc_by_year(p, y):
        df = pd.concat([p.rename("p"), y.rename("y")], axis=1).dropna()
        df = df[df["y"].isin([0, 1])]
        out = {}
        yrs = df.index.get_level_values("date").year
        for yr in sorted(set(yrs)):
            sub = df[yrs == yr]
            if sub["y"].nunique() < 2 or len(sub) < 50:
                continue
            r = sub["p"].rank()
            n1 = (sub["y"] == 1).sum(); n0 = (sub["y"] == 0).sum()
            out[int(yr)] = float((r[sub["y"] == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
        return out

    auc_year_long = _auc_by_year(p_long, y_long)
    auc_year_short = _auc_by_year(p_short, y_short)

    # --- simulate: each rebalance, enter names whose P clears P* + margin ---
    result = _simulate_barrier(
        close, p_long, p_short, y_long, y_short,
        up, down, p_star, cfg,
    )
    # SKILL-LESS BASELINE. Re-run the exact same simulation but with the
    # predicted probabilities SHUFFLED across names within each date. This earns
    # the pure geometry+drift return with zero skill. The real edge is our
    # return MINUS this baseline; comparing to zero would be dishonest because
    # the asymmetric barrier pays a skill-less book a positive return.
    rng = np.random.default_rng(cfg.model.random_state + 1)
    pl_shuf = p_long.groupby(level="date").transform(
        lambda s: pd.Series(rng.permutation(s.to_numpy()), index=s.index))
    ps_shuf = p_short.groupby(level="date").transform(
        lambda s: pd.Series(rng.permutation(s.to_numpy()), index=s.index))
    baseline = _simulate_barrier(close, pl_shuf, ps_shuf, y_long, y_short,
                                 up, down, p_star, cfg)

    edge_ann = result["stats"].get("ann_return", np.nan) - \
        baseline["stats"].get("ann_return", np.nan)

    result.update({
        "base_long": base_long, "base_short": base_short,
        "p_star": p_star, "auc_long": auc_long, "auc_short": auc_short,
        "importance": imp_long, "importance_short": imp_short,
        "up": up, "down": down,
        "auc_year_long": auc_year_long,
        "auc_year_short": auc_year_short,
        "baseline_stats": baseline["stats"],
        "baseline_gross_ann": baseline["gross_ann"],
        "edge_over_baseline_ann": float(edge_ann),
    })
    return result


def _simulate_barrier(close, p_long, p_short, y_long, y_short,
                      up, down, p_star, cfg):
    from .portfolio import performance_stats

    dates = p_long.index.get_level_values("date").unique().sort_values()
    reb = cfg.portfolio.rebalance_every
    rebal_dates = dates[::reb]
    c = cfg.costs

    n_side = cfg.portfolio.n_long
    margin = 0.02  # require P to clear P* by this much (conviction buffer)

    rows = []
    for dt in rebal_dates:
        try:
            pl = p_long.xs(dt, level="date")
            ps = p_short.xs(dt, level="date")
        except KeyError:
            continue

        # candidates that clear breakeven
        longs = pl[pl > p_star + margin].nlargest(n_side)
        shorts = ps[ps > p_star + margin].nlargest(n_side)
        shorts = shorts[~shorts.index.isin(longs.index)]

        # DRIFT NEUTRALISATION. A +up/-down barrier on a drifting universe pays
        # a positive return to RANDOM long entries (and negative to random
        # shorts) purely from beta, not skill. To isolate skill we force the
        # book market-neutral: take the SAME number of longs and shorts. Any
        # remaining P&L is cross-sectional selection, not drift harvesting.
        k_bal = min(len(longs), len(shorts))
        if k_bal == 0:
            rows.append({"date": dt, "gross": 0.0, "cost": 0.0,
                         "carry": 0.0, "net": 0.0, "n_long": 0, "n_short": 0})
            continue
        longs = longs.head(k_bal)
        shorts = shorts.head(k_bal)

        n_pos = len(longs) + len(shorts)
        w = cfg.portfolio.gross / n_pos  # equal weight per position

        # Universe drift that period: mean realised long-barrier return across
        # ALL names. Subtracting it from each long leg (and adding it back to
        # each short leg) removes the common beta/drift component, so the P&L
        # reflects SELECTION, not the fact that the market went up.
        all_r = np.array([
            realised_barrier_return(close, dt, t, up, down,
                                    cfg.label.horizon, "long")
            for t in close.columns
        ])
        drift = np.nanmean(all_r) if np.isfinite(all_r).any() else 0.0

        pnl = 0.0
        for tkr in longs.index:
            r = realised_barrier_return(close, dt, tkr, up, down,
                                        cfg.label.horizon, "long")
            if np.isfinite(r):
                pnl += w * (r - drift)          # long excess over universe
        for tkr in shorts.index:
            r = realised_barrier_return(close, dt, tkr, up, down,
                                        cfg.label.horizon, "short")
            if np.isfinite(r):
                pnl += w * (r + drift)          # short: add drift back

        cost = c.one_way * 2 * w * n_pos           # enter+exit each position
        carry = w * len(shorts) * c.borrow_annual * cfg.label.horizon / 252.0

        rows.append({"date": dt, "gross": pnl, "cost": cost, "carry": carry,
                     "net": pnl - cost - carry,
                     "n_long": len(longs), "n_short": len(shorts)})

    led = pd.DataFrame(rows).set_index("date")
    ppy = 252.0 / reb
    stats = performance_stats(led["net"], c, ppy)
    return {
        "ledger": led, "net_returns": led["net"], "stats": stats,
        "gross_ann": float(led["gross"].mean() * ppy),
        "cost_ann": float(led["cost"].mean() * ppy),
        "carry_ann": float(led["carry"].mean() * ppy),
        "avg_positions": float((led["n_long"] + led["n_short"]).mean()),
        "pct_periods_traded": float((led["n_long"] + led["n_short"] > 0).mean()),
    }
