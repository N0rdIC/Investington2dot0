"""Feature construction.

Every feature is stationary (returns, z-scores, ratios, ranks). No raw prices,
no raw volume. Non-stationary features are the most common silent killer of
ML-on-markets: the model learns "price was 40 in 2016" and generalises nothing.

Panel convention throughout: a DataFrame indexed by date, columns = tickers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ewma_vol(returns: pd.DataFrame, halflife: int = 60) -> pd.DataFrame:
    """Slow, stable volatility estimate from RETURNS (not price levels).

    This is the fix for the Bollinger denominator problem: Bollinger's sigma is
    the stdev of the last N closing *prices*, which for a trending stock mostly
    measures the trend range, not volatility.
    """
    return returns.ewm(halflife=halflife, min_periods=20).std()


def zscore_reversal(returns: pd.DataFrame, k: int, vol: pd.DataFrame) -> pd.DataFrame:
    """Cumulative k-day return expressed in units of sigma. Sign flipped so that
    a large POSITIVE value means 'fell hard' => reversal candidate (long)."""
    cum = returns.rolling(k).sum()
    return -cum / (vol * np.sqrt(k))


def sector_residual(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    k: int = 5,
    window: int = 120,
    vol_halflife: int = 60,
) -> pd.DataFrame:
    """Idiosyncratic component of the k-day move.

    Regress each stock's daily return on the factor returns over a rolling
    window, then accumulate the residual over the last k days and standardise.

    This is the operational answer to 'no news filter'. If NVDA fell 6% and SMH
    fell 6%, the residual is ~0: that is a sector repricing, not an overreaction,
    and there is nothing to fade. If NVDA fell 6% and SMH fell 1%, the residual
    is -5% and THAT is the part that mean-reverts.
    """
    common = returns.index.intersection(factors.index)
    R = returns.loc[common]
    F = factors.loc[common]

    resid = pd.DataFrame(index=R.index, columns=R.columns, dtype=float)

    Fv = F.to_numpy(dtype=float)
    n = len(R)

    for ticker in R.columns:
        y = R[ticker].to_numpy(dtype=float)
        out = np.full(n, np.nan)

        for t in range(window, n):
            sl = slice(t - window, t)          # strictly past data only
            Xw = Fv[sl]
            yw = y[sl]
            mask = np.isfinite(yw) & np.isfinite(Xw).all(axis=1)
            if mask.sum() < window // 2:
                continue
            Xd = np.column_stack([np.ones(mask.sum()), Xw[mask]])
            try:
                beta, *_ = np.linalg.lstsq(Xd, yw[mask], rcond=None)
            except np.linalg.LinAlgError:
                continue
            if not np.isfinite(Fv[t]).all() or not np.isfinite(y[t]):
                continue
            pred = beta[0] + Fv[t] @ beta[1:]
            out[t] = y[t] - pred

        resid[ticker] = out

    cum = resid.rolling(k).sum()
    rvol = resid.ewm(halflife=vol_halflife, min_periods=20).std()
    return -cum / (rvol * np.sqrt(k))


def percent_b(close: pd.DataFrame, window: int = 20, k: float = 2.0) -> pd.DataFrame:
    """Bollinger %B. Kept only as a FEATURE, never as an entry or exit rule."""
    ma = close.rolling(window).mean()
    sd = close.rolling(window).std()
    upper, lower = ma + k * sd, ma - k * sd
    return (close - lower) / (upper - lower).replace(0.0, np.nan)


def consecutive_run(returns: pd.DataFrame) -> pd.DataFrame:
    """Signed count of consecutive up/down closes. +3 = three up days in a row."""
    sign = np.sign(returns)
    out = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    prev = pd.Series(0.0, index=returns.columns)
    for dt in returns.index:
        s = sign.loc[dt]
        prev = np.where(
            (s != 0) & (np.sign(prev) == s), prev + s,
            np.where(s != 0, s, 0.0),
        )
        prev = pd.Series(prev, index=returns.columns)
        out.loc[dt] = prev
    return out


def rsi(close: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, min_periods=window).mean()
    rs = gain / loss.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def build_features(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    factor_close: pd.DataFrame,
    stress_close: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return a long-format DataFrame indexed by (date, ticker)."""
    ret = close.pct_change(fill_method=None)
    fret = factor_close.pct_change(fill_method=None)
    vol = ewma_vol(ret, halflife=60)

    feats: dict[str, pd.DataFrame] = {}

    # --- reversal family (the core hypothesis) ---
    feats["rev_1"] = zscore_reversal(ret, 1, vol)
    feats["rev_3"] = zscore_reversal(ret, 3, vol)
    feats["rev_5"] = zscore_reversal(ret, 5, vol)
    feats["rev_resid_5"] = sector_residual(ret, fret, k=5)
    feats["rev_resid_10"] = sector_residual(ret, fret, k=10)

    # --- bounce confirmation: fell hard AND has started to turn ---
    fell = feats["rev_5"] > 1.5
    turned = ret > 0
    feats["bounce"] = (fell & turned).astype(float)

    # --- band / oscillator position ---
    feats["pct_b"] = percent_b(close, 20)
    feats["rsi_14"] = rsi(close, 14) / 100.0

    # --- volume: forced selling leaves a footprint ---
    lv = np.log(volume.replace(0, np.nan))
    feats["vol_z"] = (lv - lv.rolling(20).mean()) / lv.rolling(20).std()

    # --- volatility regime: is sigma expanding? ---
    feats["rvol_ratio"] = (
        ret.rolling(5).std() / ret.rolling(60).std().replace(0.0, np.nan)
    )
    feats["rvol_60"] = ret.rolling(60).std()

    # --- trend context (the load-bearing filter, as a feature) ---
    ma200 = close.rolling(200).mean()
    feats["dist_ma200"] = close / ma200 - 1.0
    feats["ma200_slope"] = ma200.pct_change(20)

    # --- medium-term momentum, skipping the last week to avoid overlap ---
    feats["mom_60"] = close.shift(5) / close.shift(65) - 1.0

    # --- DEEPER HISTORY (point 4): quarterly / annual / multi-year context ---
    # All skip the last 5 days so they never overlap the label window.
    # NaN for names younger than the lookback; XGBoost routes NaN natively, so
    # young names simply don't use the deep features rather than being dropped.
    feats["mom_21"] = close.shift(5) / close.shift(26) - 1.0       # ~1 month
    feats["mom_126"] = close.shift(5) / close.shift(131) - 1.0     # ~6 months
    feats["mom_252"] = close.shift(5) / close.shift(257) - 1.0     # ~1 year
    feats["mom_756"] = close.shift(5) / close.shift(761) - 1.0     # ~3 years
    feats["mom_1260"] = close.shift(5) / close.shift(1265) - 1.0   # ~5 years

    # position within the multi-year range (0 = 3y low, 1 = 3y high)
    roll_max = close.rolling(756, min_periods=120).max()
    roll_min = close.rolling(756, min_periods=120).min()
    feats["range_pos_3y"] = (close - roll_min) / (roll_max - roll_min).replace(0, np.nan)

    # long-horizon realised vol and its ratio to short-horizon (regime)
    feats["rvol_252"] = ret.rolling(252, min_periods=60).std()
    feats["rvol_ratio_long"] = (
        ret.rolling(20).std() / ret.rolling(252, min_periods=60).std().replace(0, np.nan)
    )

    # drawdown from trailing 1y peak (how deep in a hole is it?)
    peak_1y = close.rolling(252, min_periods=60).max()
    feats["drawdown_1y"] = close / peak_1y - 1.0

    # annual-scale momentum on the sector residual is expensive; use raw here
    feats["mom_252_vol_adj"] = feats["mom_252"] / (vol * np.sqrt(252)).replace(0, np.nan)

    # --- streaks and gaps ---
    feats["run"] = consecutive_run(ret)
    feats["gap"] = (close / close.shift(1) - 1.0) / vol

    # =====================================================================
    #  NEW FEATURE FAMILIES (creative extensions)
    # =====================================================================

    # --- (a) RETURN DISTRIBUTION SHAPE: skew & kurtosis of recent returns ---
    # Crashes and melt-ups have different higher moments. A name whose 60-day
    # return skew is very negative has been taking the stairs up, elevator down.
    feats["skew_60"] = ret.rolling(60).skew()
    feats["kurt_60"] = ret.rolling(60).kurt()
    feats["skew_20"] = ret.rolling(20).skew()

    # --- (b) VOL-OF-VOL: is the volatility itself unstable? ---
    # Regime transitions show up as rising vol-of-vol before rising vol.
    dvol = ret.rolling(10).std()
    feats["vol_of_vol"] = dvol.rolling(60).std() / dvol.rolling(60).mean().replace(0, np.nan)

    # --- (c) ACCELERATION: 2nd derivative of price (momentum of momentum) ---
    m5 = close.pct_change(5)
    feats["accel"] = (m5 - m5.shift(5)) / vol.replace(0, np.nan)

    # --- (d) DOWNSIDE vs UPSIDE vol asymmetry (Sortino-style) ---
    neg = ret.where(ret < 0)
    pos = ret.where(ret > 0)
    dn = neg.rolling(60, min_periods=15).std()
    upv = pos.rolling(60, min_periods=15).std()
    feats["vol_asym"] = (dn - upv) / (dn + upv).replace(0, np.nan)

    # --- (e) LIQUIDITY / Amihud illiquidity: |return| per unit dollar volume ---
    dollar_vol = (close * volume).replace(0, np.nan)
    feats["amihud"] = (ret.abs() / dollar_vol).rolling(20).mean() * 1e9
    feats["dollar_vol_z"] = (
        np.log(dollar_vol) - np.log(dollar_vol).rolling(60).mean()
    ) / np.log(dollar_vol).rolling(60).std()

    # --- (f) MAX-return factor (lottery demand): largest 1-day gain in 20d ---
    # High "MAX" names are lottery-like and tend to underperform (Bali et al).
    feats["max_ret_20"] = ret.rolling(20).max()
    feats["min_ret_20"] = ret.rolling(20).min()

    # --- (g) 52-week-high proximity (anchoring effect) ---
    high_252 = close.rolling(252, min_periods=60).max()
    feats["near_52w_high"] = close / high_252

    # --- (h) SEASONALITY: day-of-week and turn-of-month effects ---
    dow = pd.Series(close.index.dayofweek, index=close.index)
    dom = pd.Series(close.index.day, index=close.index)
    tom = ((dom >= 28) | (dom <= 2)).astype(float)       # turn of month
    # broadcast to every ticker
    feats["is_monday"] = pd.DataFrame(
        np.repeat((dow == 0).to_numpy()[:, None], close.shape[1], axis=1),
        index=close.index, columns=close.columns, dtype=float)
    feats["turn_of_month"] = pd.DataFrame(
        np.repeat(tom.to_numpy()[:, None], close.shape[1], axis=1),
        index=close.index, columns=close.columns, dtype=float)

    # --- (i) REGIME / STRESS features (same for all names on a given date) ---
    # These let the model CONDITION on crisis regimes (2008, 2020, 2022) even
    # for names that didn't exist in 2008. Derived from SPY so they reach back
    # as far as SPY does (1993).
    if stress_close is not None and "SPY" in stress_close.columns:
        spy = stress_close["SPY"]
        spy_ret = spy.pct_change(fill_method=None)
        # synthetic VIX-like stress: annualised 20d realized vol of SPY
        stress = spy_ret.rolling(20).std() * np.sqrt(252)
        spy_dd = spy / spy.rolling(252, min_periods=60).max() - 1.0
        stress_z = (stress - stress.rolling(252, min_periods=60).mean()) / \
            stress.rolling(252, min_periods=60).std()

        def _bcast(s):
            return pd.DataFrame(
                np.repeat(s.to_numpy()[:, None], close.shape[1], axis=1),
                index=close.index, columns=close.columns, dtype=float
            ).reindex(close.index)

        feats["mkt_stress"] = _bcast(stress.reindex(close.index))
        feats["mkt_stress_z"] = _bcast(stress_z.reindex(close.index))
        feats["mkt_drawdown"] = _bcast(spy_dd.reindex(close.index))

        # credit stress: HYG/TLT ratio (risk-on vs risk-off), where available
        if "HYG" in stress_close.columns and "TLT" in stress_close.columns:
            credit = (stress_close["HYG"].pct_change(20)
                      - stress_close["TLT"].pct_change(20))
            feats["credit_stress"] = _bcast(credit.reindex(close.index))
        # gold: crisis hedge behaviour
        if "GLD" in stress_close.columns:
            gld_mom = stress_close["GLD"].pct_change(20)
            feats["gold_mom"] = _bcast(gld_mom.reindex(close.index))
        # dollar: risk-off proxy
        if "UUP" in stress_close.columns:
            usd_mom = stress_close["UUP"].pct_change(20)
            feats["usd_mom"] = _bcast(usd_mom.reindex(close.index))

        # --- (j) BETA TO MARKET STRESS: does this name blow up in crises? ---
        # 120-day rolling beta of the name to SPY; high-beta names behave very
        # differently across regimes, and this interacts with mkt_stress.
        m = spy_ret.reindex(ret.index)
        cov = ret.rolling(120).cov(m)
        var = m.rolling(120).var()
        feats["beta_spy"] = cov.div(var, axis=0)

    panel = pd.concat(
        {name: df.stack(future_stack=True) for name, df in feats.items()},
        axis=1,
    )
    panel.index.names = ["date", "ticker"]
    return panel


def cross_sectional_zscore(panel: pd.DataFrame) -> pd.DataFrame:
    """Normalise each feature within each date.

    This is what makes the problem cross-sectional. The model is being asked
    'which of these 30 names is most attractive TODAY', not 'is the market
    going up', so absolute levels must be removed date by date.
    """
    g = panel.groupby(level="date")
    mu = g.transform("mean")
    sd = g.transform("std")
    # A zero-variance cross-section (e.g. a binary flag that is 0 for every name
    # today) must map to 0, NOT NaN. Mapping it to NaN silently deletes the whole
    # date downstream -- this bug cost us 80% of our trading days.
    z = (panel - mu) / sd.where(sd > 1e-12)
    z = z.where(sd.notna() & (sd > 1e-12), 0.0)
    return z.clip(-4.0, 4.0)


def make_labels(close: pd.DataFrame, horizon: int = 5, demean: bool = True) -> pd.Series:
    """Forward `horizon`-day return, cross-sectionally demeaned.

    Demeaning is what converts this from a (hopeless) market-timing problem into
    a (tractable) ranking problem, and it makes the resulting book dollar-neutral
    by construction.
    """
    fwd = close.shift(-horizon) / close - 1.0
    y = fwd.stack(future_stack=True)
    y.index.names = ["date", "ticker"]
    if demean:
        y = y - y.groupby(level="date").transform("mean")
    return y.rename("y")
