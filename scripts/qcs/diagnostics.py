"""Diagnostics.

These matter more than the headline Sharpe. A backtest reporting Sharpe 1.5 on
30 correlated names over 10 years, where every good trade happened in the same
three months, is not a strategy - it is a single bet dressed up as 500.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def effective_n(returns: pd.DataFrame, weights_history: pd.DataFrame | None = None) -> dict:
    """How many INDEPENDENT bets are you actually making?

    Your nominal N is (number of rebalances) x (positions per rebalance). But if
    all six crypto-proxies move together and all seven semis move together, your
    effective N is a small fraction of that - and every confidence interval in
    your backtest is too narrow by sqrt(N_nominal / N_effective).

    Uses the participation ratio of the correlation matrix eigenvalues:
        N_eff = (sum lambda)^2 / sum(lambda^2)
    """
    C = returns.corr().dropna(how="all").dropna(axis=1, how="all")
    if C.empty:
        return {}
    eig = np.linalg.eigvalsh(C.to_numpy())
    eig = eig[eig > 1e-10]
    n_eff = (eig.sum() ** 2) / (eig ** 2).sum()

    return {
        "n_names": int(C.shape[0]),
        "n_effective_names": float(n_eff),
        "collapse_ratio": float(n_eff / C.shape[0]),
        "mean_pairwise_corr": float(
            C.to_numpy()[np.triu_indices_from(C.to_numpy(), k=1)].mean()
        ),
        "top_eigenvalue_share": float(eig.max() / eig.sum()),
    }


def pnl_concentration(net: pd.Series) -> dict:
    """Is the P&L coming from everywhere, or from three lucky weeks?"""
    net = net.dropna()
    if len(net) == 0:
        return {}

    total = net.sum()
    top5 = net.nlargest(max(1, len(net) // 20)).sum()

    # Gini-like measure of how concentrated the positive P&L is
    pos = net[net > 0].sort_values(ascending=False)
    share_top_decile = (
        pos.head(max(1, len(pos) // 10)).sum() / pos.sum() if len(pos) else np.nan
    )

    return {
        "total_return_sum": float(total),
        "top_5pct_periods_share": float(top5 / total) if total != 0 else np.nan,
        "top_decile_of_winners_share": float(share_top_decile),
        "worst_period": float(net.min()),
        "best_period": float(net.max()),
    }


def cost_decomposition(ledger: pd.DataFrame, periods_per_year: float) -> pd.DataFrame:
    """Where the money goes. This is usually the whole story."""
    rows = {
        "gross alpha": ledger["gross"].mean() * periods_per_year,
        "transaction cost": -ledger["tcost"].mean() * periods_per_year,
        "borrow / carry": -ledger["carry"].mean() * periods_per_year,
    }
    rows["net (pre-tax)"] = sum(rows.values())
    df = pd.DataFrame.from_dict(rows, orient="index", columns=["annualised"])
    df["pct_of_gross"] = df["annualised"] / rows["gross alpha"] * 100
    return df


def breakeven_ic(
    costs_ann: float,
    ic: float,
    gross_ann: float,
) -> float:
    """What IC would you need for the strategy to break even on costs?

    Gross alpha scales roughly linearly with IC, so:
        ic_breakeven = ic_observed * (costs_ann / gross_ann)
    """
    if gross_ann <= 0:
        return np.nan
    return ic * costs_ann / gross_ann


def report(result: dict, close: pd.DataFrame, cfg) -> str:
    """Human-readable summary."""
    ppy = 252.0 / cfg.portfolio.rebalance_every
    ret = close.pct_change(fill_method=None)

    s = result["stats"]
    ics = result["ic_summary"]
    eff = effective_n(ret)
    conc = pnl_concentration(result["net_returns"])
    costs = cost_decomposition(result["ledger"], ppy)

    total_cost = result["tcost_ann"] + result["carry_ann"]
    be_ic = breakeven_ic(total_cost, ics["ic_mean"], result["gross_ann"])

    lines = []
    lines.append("=" * 66)
    lines.append("SIGNAL QUALITY")
    lines.append("=" * 66)
    lines.append(f"  mean IC (rank corr)        {ics['ic_mean']:+.4f}")
    lines.append(f"  IC t-stat (overlap-adj)    {ics['ic_t_stat']:+.2f}")
    lines.append(f"  IC hit rate                {ics['ic_hit_rate']:.1%}")
    lines.append(f"  effective observations     {ics['n_effective']:.0f}"
                 f"  (nominal {ics['n_obs']})")
    lines.append("")
    lines.append("=" * 66)
    lines.append(f"COST DECOMPOSITION  (c = {cfg.costs.round_trip:.2%} round trip)")
    lines.append("=" * 66)
    for name, row in costs.iterrows():
        lines.append(f"  {name:<24} {row['annualised']:+8.2%}")
    lines.append(f"  {'-' * 40}")
    lines.append(f"  avg turnover / rebalance   {result['avg_turnover']:.2f}"
                 f"  (2.00 = full replacement)")
    lines.append(f"  breakeven IC               {be_ic:.4f}"
                 f"   (you have {ics['ic_mean']:.4f})")
    lines.append("")
    lines.append("=" * 66)
    lines.append("PERFORMANCE")
    lines.append("=" * 66)
    if s:
        lines.append(f"  net return (ann)           {s['ann_return']:+.2%}")
        lines.append(f"  volatility (ann)           {s['ann_vol']:.2%}")
        lines.append(f"  Sharpe                     {s['sharpe']:+.2f}")
        lines.append(f"  max drawdown               {s['max_drawdown']:.2%}")
        lines.append(f"  t-stat on mean return      {s['t_stat']:+.2f}")
        lines.append(f"  after 30% flat tax         {s['after_tax_return']:+.2%}")
    lines.append("")
    lines.append("=" * 66)
    lines.append("INDEPENDENCE / CONCENTRATION")
    lines.append("=" * 66)
    lines.append(f"  names in universe          {eff['n_names']}")
    lines.append(f"  EFFECTIVE independent      {eff['n_effective_names']:.1f}"
                 f"   ({eff['collapse_ratio']:.0%} of nominal)")
    lines.append(f"  mean pairwise correlation  {eff['mean_pairwise_corr']:.2f}")
    lines.append(f"  1st eigenvalue share       {eff['top_eigenvalue_share']:.0%}")
    lines.append(f"  P&L from top 5% of periods {conc['top_5pct_periods_share']:.0%}")
    lines.append("")
    lines.append("TOP FEATURES")
    lines.append("-" * 66)
    for name, val in result["importance"].head(8).items():
        lines.append(f"  {name:<24} {val:.3f}")

    return "\n".join(lines)


def ic_decay(pred: pd.Series, close: pd.DataFrame, horizons=(1, 2, 3, 5, 10, 20, 40)) -> pd.DataFrame:
    """How long does the edge actually persist?

    THIS DIAGNOSTIC SETS YOUR HOLDING PERIOD, not the cost formula.

    The cost optimum h* = (2c / e*sigma)^2 assumes your expected move grows as
    sqrt(h) -- i.e. that the signal predicts the WHOLE h-day move. It does not.
    A signal that decays in 5 days, held for 20, gives you 5 days of edge and 15
    days of noise. Gross alpha stops growing; costs keep falling; the formula
    happily recommends a horizon at which you have no signal left.

    So: measure the decay FIRST. Then, among horizons where IC is still alive,
    pick the one the cost model prefers. If the cost-preferred h lies beyond the
    decay horizon, the strategy is not viable at that cost -- full stop.
    """
    from .cv import information_coefficient

    rows = []
    for h in horizons:
        fwd = close.shift(-h) / close - 1.0
        y = fwd.stack(future_stack=True)
        y.index.names = ["date", "ticker"]
        y = y - y.groupby(level="date").transform("mean")
        ic = information_coefficient(pred, y.reindex(pred.index))
        n_eff = max(1.0, len(ic) / h)
        mean, sd = float(ic.mean()), float(ic.std(ddof=1))
        rows.append({
            "horizon_d": h,
            "ic": mean,
            "ic_t": mean / (sd / np.sqrt(n_eff)) if sd > 0 else np.nan,
            # cumulative edge available if held h days, in IC-units
            "edge_proxy": mean * np.sqrt(h),
        })
    return pd.DataFrame(rows).set_index("horizon_d")
