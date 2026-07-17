"""Triple-barrier labels (Lopez de Prado).

This replaces the forward-return regression target with your PATH-DEPENDENT
event: within `horizon` days, does the stock touch the profit barrier (+G)
BEFORE it touches the stop barrier (-L)?

    label = 1  if +G is hit before -L within the window     (event happened)
    label = 0  otherwise (stop hit first, or neither hit)

This is exactly the G / L / P framing from the very first message:
    G = up_barrier, L = down_barrier, P = P(label = 1)
and the model's predicted probability IS your P.

Why path dependence matters: a plain 5-day forward return of +2% could have
dipped -8% mid-week first (blowing a -3% stop). The barrier label knows the
difference; a forward-return label is blind to it. This is the single most
important correction to the earlier design.

The SHORT book uses the mirror label: does -G_short trigger before +L_short.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def triple_barrier_labels(
    close: pd.DataFrame,
    up: float = 0.06,
    down: float = 0.03,
    horizon: int = 5,
    side: str = "long",
) -> pd.Series:
    """Return a 0/1 label per (date, ticker).

    side='long':  label 1 if price rises +up before falling -down within horizon.
    side='short': label 1 if price falls -up before rising +down within horizon
                  (i.e. a profitable short of the same G/L geometry).

    Barriers are measured on the raw price path using intraday-free close data,
    so a barrier counts as touched on day t if that day's CLOSE breaches it.
    (Using closes is conservative: real intraday highs/lows would trigger more
    often, but daily OHLC isn't loaded here and closes avoid look-ahead.)
    """
    if side not in ("long", "short"):
        raise ValueError("side must be 'long' or 'short'")

    px = close.to_numpy(dtype=float)
    n, k = px.shape
    out = np.full((n, k), np.nan)

    # For each ticker, build the forward-return matrix over the horizon window
    # and find which barrier is touched FIRST. Vectorised over time within the
    # horizon (small constant loop of length `horizon`), not over all days.
    for j in range(k):
        p = px[:, j]
        valid0 = np.isfinite(p) & (p > 0)

        # first-touch day for up and down barrier (inf if never within horizon)
        up_day = np.full(n, np.inf)
        dn_day = np.full(n, np.inf)

        for h in range(1, horizon + 1):
            fp = np.full(n, np.nan)
            fp[: n - h] = p[h:]
            r = fp / p - 1.0  # h-day forward return from each t

            if side == "long":
                up_hit = r >= up
                dn_hit = r <= -down
            else:
                up_hit = r <= -up
                dn_hit = r >= down

            newer_up = up_hit & np.isfinite(r) & (h < up_day)
            up_day = np.where(newer_up, h, up_day)
            newer_dn = dn_hit & np.isfinite(r) & (h < dn_day)
            dn_day = np.where(newer_dn, h, dn_day)

        # event = up barrier strictly before down barrier
        lab_col = np.where(up_day < dn_day, 1.0, 0.0)
        # if neither barrier ever touched, still label 0 (event did not happen),
        # but require the start point to be valid
        out[:, j] = np.where(valid0, lab_col, np.nan)
        out[n - 1, j] = np.nan  # last day has no forward window

    lab = pd.DataFrame(out, index=close.index, columns=close.columns)
    s = lab.stack(future_stack=True)
    s.index.names = ["date", "ticker"]
    return s.rename("y")


def barrier_base_rate(labels: pd.Series) -> dict:
    """Unconditional P(event) - the base rate the model must beat.

    If +6/-3 fires 33% of the time unconditionally, a model predicting the
    constant 0.33 has zero skill. Skill = ranking days ABOVE vs BELOW this rate.
    Also reports the breakeven P* implied by the barrier geometry.
    """
    y = labels.dropna()
    return {
        "base_rate": float(y.mean()),
        "n_labels": int(len(y)),
        "n_events": int(y.sum()),
    }
