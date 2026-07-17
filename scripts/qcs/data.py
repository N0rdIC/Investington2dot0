"""Data loading.

load_yahoo() runs on YOUR machine (this sandbox has no market-data network
access). synth() generates controlled panels used to validate the backtest
machinery before it is ever pointed at real money.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd


def load_yahoo(tickers, start, end, cache="cache"):
    """Download adjusted OHLCV via yfinance.

        pip install yfinance

    NOTE ON SURVIVORSHIP BIAS: this pulls today's tickers, which are in your
    universe partly BECAUSE they survived and did well. A backtest on such a
    universe is biased upward and there is no way to fix it with this data
    source. To do it properly you need a point-in-time universe (e.g. CRSP,
    Norgate, or Sharadar). Treat any result from Yahoo data as an OPTIMISTIC
    upper bound, not an estimate.
    """
    import yfinance as yf

    os.makedirs(cache, exist_ok=True)
    key = os.path.join(cache, f"{hash((tuple(tickers), start, end)) & 0xFFFFFFF}.pkl")
    if os.path.exists(key):
        return pd.read_pickle(key)

    raw = yf.download(
        list(tickers), start=start, end=end,
        auto_adjust=True, progress=False, group_by="column",
    )
    close = raw["Close"].dropna(how="all")
    volume = raw["Volume"].reindex(close.index)

    out = {"close": close, "volume": volume}
    pd.to_pickle(out, key)
    return out


def synth(
    n_days: int = 2600,
    tickers=None,
    factors=None,
    daily_vol: float = 0.035,
    market_vol: float = 0.011,
    beta_mean: float = 1.4,
    reversal_strength: float = 0.0,
    horizon: int = 5,
    seed: int = 7,
):
    """Synthetic panel with a KNOWN, INJECTED cross-sectional reversal effect.

    Construction:
      r[t,i] = beta[i] * market[t] + idio[t,i]
      idio[t,i] = -reversal_strength * z(past 5d idio) * sigma + noise

    `reversal_strength` is the ground truth. Set it to 0.0 for a pure random
    walk (the NULL) and to e.g. 0.04 to plant a modest, realistic effect.

    This lets us ask the only question that matters about a backtest:
    DOES IT REPORT THE TRUTH? If it finds signal when reversal_strength=0, it
    leaks. If it finds nothing when reversal_strength=0.04, it is blind.
    """
    rng = np.random.default_rng(seed)

    tickers = tickers or [f"S{i:02d}" for i in range(30)]
    factors = factors or ["MKT", "SEC1", "SEC2", "SEC3"]
    n, k = n_days, len(tickers)

    dates = pd.bdate_range("2014-01-02", periods=n)

    market = rng.normal(0.0004, market_vol, n)
    sector = rng.normal(0.0, market_vol * 0.7, (n, 3))
    sector_map = rng.integers(0, 3, k)

    beta = rng.normal(beta_mean, 0.35, k).clip(0.5, 2.5)
    sbeta = rng.normal(1.0, 0.3, k).clip(0.2, 2.0)

    idio = np.zeros((n, k))
    ret = np.zeros((n, k))
    sig = daily_vol

    for t in range(n):
        shock = rng.normal(0.0, sig, k)

        if t >= horizon and reversal_strength > 0:
            past = idio[t - horizon: t].sum(axis=0)
            z = past / (sig * np.sqrt(horizon))
            shock = shock - reversal_strength * np.clip(z, -3, 3) * sig

        idio[t] = shock
        ret[t] = beta * market[t] + sbeta * sector[t, sector_map] + idio[t]

    close = pd.DataFrame(
        100.0 * np.exp(np.cumsum(ret, axis=0)), index=dates, columns=tickers
    )

    fret = np.column_stack([market, sector])
    factor_close = pd.DataFrame(
        100.0 * np.exp(np.cumsum(fret, axis=0)), index=dates, columns=factors
    )

    base = rng.lognormal(15.5, 0.4, k)
    vshock = rng.lognormal(0.0, 0.5, (n, k))
    # volume spikes with the size of the move (forced-selling footprint)
    volume = pd.DataFrame(
        base * vshock * (1.0 + 3.0 * np.abs(ret) / sig * 0.3),
        index=dates, columns=tickers,
    )

    return {"close": close, "volume": volume, "factor_close": factor_close}
