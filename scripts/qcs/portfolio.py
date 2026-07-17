"""Portfolio construction and cost accounting.

THE CENTRAL ARITHMETIC OF THIS FILE
-----------------------------------
Cost per rebalance = one_way * sum(|delta_weight|)

With c = 0.20% round trip, one_way = 0.10%. A weekly rebalance that fully
replaces the book moves sum(|dw|) = 2.0 (close everything, open everything at
100% gross), so:

    cost = 0.0010 * 2.0 = 0.20% PER WEEK  ->  ~10.4% PER YEAR

That is an enormous hurdle. The hysteresis buffer below exists to attack it:
an existing long is held until its rank falls out of the top `buffer_rank`,
rather than being dumped the moment it leaves the top `n_long`. In practice this
roughly halves turnover for a small loss of signal purity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Costs, PortfolioCfg


def target_weights(
    scores: pd.Series,
    prev_w: pd.Series,
    cfg: PortfolioCfg,
    betas: pd.Series | None = None,
) -> pd.Series:
    """Map cross-sectional scores to dollar-neutral target weights.

    Long the top `n_long`, short the bottom `n_short`, equal weight, with a
    hysteresis buffer to suppress turnover.
    """
    scores = scores.dropna()
    if len(scores) < cfg.n_long + cfg.n_short:
        return pd.Series(0.0, index=prev_w.index)

    rank_desc = scores.rank(ascending=False)   # 1 = best (most attractive long)
    rank_asc = scores.rank(ascending=True)     # 1 = worst (most attractive short)

    prev_long = set(prev_w[prev_w > 0].index)
    prev_short = set(prev_w[prev_w < 0].index)

    # --- LONG book ---
    keep_long = {t for t in prev_long
                 if t in rank_desc.index and rank_desc[t] <= cfg.buffer_rank}
    longs = list(keep_long)
    for t in rank_desc.sort_values().index:
        if len(longs) >= cfg.n_long:
            break
        if t not in longs:
            longs.append(t)
    longs = longs[: cfg.n_long]

    # --- SHORT book ---
    shorts: list[str] = []
    if cfg.allow_short:
        keep_short = {t for t in prev_short
                      if t in rank_asc.index and rank_asc[t] <= cfg.buffer_rank}
        shorts = [t for t in keep_short if t not in longs]
        for t in rank_asc.sort_values().index:
            if len(shorts) >= cfg.n_short:
                break
            if t not in shorts and t not in longs:
                shorts.append(t)
        shorts = shorts[: cfg.n_short]

    w = pd.Series(0.0, index=scores.index)

    # DOLLAR-NEUTRAL IS NOT BETA-NEUTRAL. With betas spanning 0.5-2.5, an equal
    # dollar long/short book can carry large net market exposure -- which is the
    # opposite of what a "market-neutral" strategy is for. Size the two sides so
    # that sum(w_i * beta_i) = 0.
    if betas is not None and longs and shorts:
        bL = float(np.nanmean(betas.reindex(longs).to_numpy()))
        bS = float(np.nanmean(betas.reindex(shorts).to_numpy()))
        if np.isfinite(bL) and np.isfinite(bS) and (bL + bS) > 1e-6:
            side_L = cfg.gross * bS / (bL + bS)
            side_S = cfg.gross * bL / (bL + bS)
        else:
            side_L = side_S = cfg.gross / 2.0
    else:
        side_L = side_S = cfg.gross / 2.0

    if longs:
        w[longs] = side_L / len(longs)
    if shorts:
        w[shorts] = -side_S / len(shorts)

    return w.reindex(prev_w.index.union(w.index)).fillna(0.0)


def rebalance_cost(w_new: pd.Series, w_old: pd.Series, costs: Costs) -> float:
    """Transaction cost of moving from w_old to w_new, as a fraction of capital."""
    idx = w_new.index.union(w_old.index)
    dw = w_new.reindex(idx).fillna(0.0) - w_old.reindex(idx).fillna(0.0)
    return costs.one_way * float(dw.abs().sum())


def carry_cost(w: pd.Series, costs: Costs, days: int) -> float:
    """Borrow / financing cost, charged on SHORT notional only."""
    short_notional = float(w[w < 0].abs().sum())
    return short_notional * costs.borrow_annual * days / 252.0


def turnover(w_new: pd.Series, w_old: pd.Series) -> float:
    idx = w_new.index.union(w_old.index)
    dw = w_new.reindex(idx).fillna(0.0) - w_old.reindex(idx).fillna(0.0)
    return float(dw.abs().sum())


def performance_stats(
    rets: pd.Series,
    costs: Costs,
    periods_per_year: float,
) -> dict:
    """Annualised statistics. Reports gross, net-of-cost, and net-of-tax."""
    rets = rets.dropna()
    if len(rets) < 2:
        return {}

    mean = float(rets.mean())
    sd = float(rets.std(ddof=1))

    ann_ret = mean * periods_per_year
    ann_vol = sd * np.sqrt(periods_per_year)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan

    equity = (1.0 + rets).cumprod()
    dd = equity / equity.cummax() - 1.0

    # t-stat on the mean return: is this distinguishable from zero at all?
    t_stat = mean / (sd / np.sqrt(len(rets))) if sd > 0 else np.nan

    return {
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": float(dd.min()),
        "hit_rate": float((rets > 0).mean()),
        "n_periods": len(rets),
        "t_stat": t_stat,
        "after_tax_return": ann_ret * (1 - costs.flat_tax) if ann_ret > 0 else ann_ret,
    }
