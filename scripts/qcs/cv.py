"""Purged, embargoed walk-forward cross-validation.

WHY THIS EXISTS
---------------
Standard k-fold CV leaks catastrophically on financial panels, for two reasons:

1. OVERLAPPING LABELS. A 5-day forward return observed on Monday shares four of
   its five days with Tuesday's label. If Monday lands in train and Tuesday in
   test, the model has already seen 80% of the answer. Fix: PURGE - drop any
   training sample whose label window overlaps the test window.

2. AUTOCORRELATED FEATURES. A 60-day EWMA volatility straddles the train/test
   boundary: the last training row and the first test row are built from almost
   the same data. Fix: EMBARGO - drop an additional buffer of days after the
   test block.

Skip either one and your reported IC will be beautiful and completely fake.
This is the single most common reason an XGBoost equity model backtests at
Sharpe 3 and then loses money in production.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class PurgedWalkForward:
    """Expanding-window walk-forward with purging and embargo.

    Yields (train_dates, test_dates) as arrays of unique dates.
    """

    def __init__(
        self,
        n_splits: int = 6,
        purge: int = 5,
        embargo: int = 10,
        min_train: int = 500,
    ):
        self.n_splits = n_splits
        self.purge = purge
        self.embargo = embargo
        self.min_train = min_train

    def split(self, dates: pd.DatetimeIndex):
        dates = pd.DatetimeIndex(sorted(pd.unique(dates)))
        n = len(dates)

        if n <= self.min_train + self.n_splits * 20:
            raise ValueError(
                f"Not enough history: {n} days for min_train={self.min_train} "
                f"and {self.n_splits} splits."
            )

        # Carve the post-min_train tail into n_splits contiguous test blocks.
        test_start_idx = self.min_train
        remaining = n - test_start_idx
        block = remaining // self.n_splits

        for i in range(self.n_splits):
            t0 = test_start_idx + i * block
            t1 = t0 + block if i < self.n_splits - 1 else n

            test_dates = dates[t0:t1]

            # Train on everything before the test block, MINUS the purge window.
            # A training sample dated d has a label spanning [d, d + horizon].
            # It overlaps the test block if d + purge >= test_start.
            # So we must drop the last `purge` training days.
            train_end_idx = max(0, t0 - self.purge)
            train_dates = dates[:train_end_idx]

            if len(train_dates) < self.min_train // 2:
                continue

            yield train_dates, test_dates

    def describe(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        rows = []
        for i, (tr, te) in enumerate(self.split(dates)):
            rows.append(
                {
                    "fold": i,
                    "train_start": tr[0].date(),
                    "train_end": tr[-1].date(),
                    "n_train_days": len(tr),
                    "purged_days": self.purge,
                    "test_start": te[0].date(),
                    "test_end": te[-1].date(),
                    "n_test_days": len(te),
                }
            )
        return pd.DataFrame(rows)


def information_coefficient(pred: pd.Series, y: pd.Series) -> pd.Series:
    """Per-date Spearman rank correlation between prediction and realised return.

    IC is the honest scorecard for a cross-sectional ranker. It is also, to a
    good approximation, the PER-TRADE SHARPE - the quantity called `e` in the
    cost model. IC in 0.02-0.05 is a genuinely good result. IC > 0.10 on a
    retail feature set almost certainly means you have a leak.
    """
    df = pd.concat([pred.rename("p"), y.rename("y")], axis=1).dropna()

    def _ic(g):
        if len(g) < 5:
            return np.nan
        return g["p"].rank().corr(g["y"].rank())

    return df.groupby(level="date").apply(_ic).dropna()


def ic_summary(ic: pd.Series, horizon: int = 5) -> dict:
    """Mean IC, its standard error, and the t-stat.

    The t-stat is what tells you whether the IC is real or luck. Because labels
    overlap, the effective number of independent observations is roughly
    len(ic) / horizon - NOT len(ic). Ignoring this inflates t by sqrt(horizon)
    (about 2.2x at a 5-day horizon), which is exactly how a worthless model
    passes a significance test.
    """
    ic = ic.dropna()
    n_eff = max(1.0, len(ic) / horizon)
    mean = float(ic.mean())
    sd = float(ic.std(ddof=1)) if len(ic) > 1 else np.nan
    se = sd / np.sqrt(n_eff) if sd and np.isfinite(sd) else np.nan
    return {
        "ic_mean": mean,
        "ic_std": sd,
        "n_obs": len(ic),
        "n_effective": n_eff,
        "ic_t_stat": mean / se if se and se > 0 else np.nan,
        "ic_hit_rate": float((ic > 0).mean()),
    }
