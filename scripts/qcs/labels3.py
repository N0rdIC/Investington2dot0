"""Joint 3-class barrier labels.

THE PROBLEM THIS SOLVES
-----------------------
Training two independent binary models (one for the long barrier, one for the
short) lets them contradict each other: both can output a high probability for
the same name on the same day. That is not merely odd, it is IMPOSSIBLE --
if +6% is touched before -3%, the path crossed +3% on the way up, which is the
short's stop, so the short outcome cannot also have occurred.

Two separate softmaxes have no way to know this. One joint softmax does.

THE CLASS DEFINITION
--------------------
Given the long bracket {+up, -down} and the mirror short bracket {-up, +down},
every price path within the horizon falls into exactly one of:

    class 2  LONG_WIN   : +up  touched before -down
    class 0  SHORT_WIN  : -up  touched before +down
    class 1  NEITHER    : anything else (stopped out either way, or timed out)

These are mutually exclusive and exhaustive -- verified empirically in
verify_exhaustive() below. A single multiclass model over them satisfies
    p_long + p_short + p_neither = 1
by construction, so the contradiction becomes inexpressible rather than merely
filtered out after the fact.

TRADING INTERPRETATION
----------------------
    p_long  = P(class 2) = P(win) for a long entry with this bracket
    p_short = P(class 0) = P(win) for a short entry with the mirror bracket
    tilt    = p_long - p_short   (now genuinely bounded, since they sum <= 1)

The breakeven test is unchanged: enter long when p_long > P* = (L+c)/(G+L).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SHORT_WIN, NEITHER, LONG_WIN = 0, 1, 2


def joint_barrier_labels(
    close: pd.DataFrame,
    up: float = 0.06,
    down: float = 0.03,
    horizon: int = 5,
) -> pd.Series:
    """Return a 0/1/2 class label per (date, ticker).

    Barriers are evaluated on closing prices, day by day forward, and the FIRST
    barrier touched decides the class. Using closes (not intraday high/low) is
    deliberate: it matches the data we have and avoids look-ahead.
    """
    px = close.to_numpy(dtype=float)
    n, k = px.shape
    out = np.full((n, k), np.nan)

    for j in range(k):
        p = px[:, j]
        valid0 = np.isfinite(p) & (p > 0)

        # first day (within horizon) each of the four levels is touched
        d_up_big = np.full(n, np.inf)    # +up   -> long profit
        d_dn_sml = np.full(n, np.inf)    # -down -> long stop
        d_dn_big = np.full(n, np.inf)    # -up   -> short profit
        d_up_sml = np.full(n, np.inf)    # +down -> short stop

        for h in range(1, horizon + 1):
            fp = np.full(n, np.nan)
            fp[: n - h] = p[h:]
            r = fp / p - 1.0
            ok = np.isfinite(r)

            d_up_big = np.where(ok & (r >= up) & (h < d_up_big), h, d_up_big)
            d_dn_sml = np.where(ok & (r <= -down) & (h < d_dn_sml), h, d_dn_sml)
            d_dn_big = np.where(ok & (r <= -up) & (h < d_dn_big), h, d_dn_big)
            d_up_sml = np.where(ok & (r >= down) & (h < d_up_sml), h, d_up_sml)

        long_win = d_up_big < d_dn_sml     # +up before -down
        short_win = d_dn_big < d_up_sml    # -up before +down

        col = np.full(n, float(NEITHER))
        col[long_win] = LONG_WIN
        col[short_win] = SHORT_WIN        # cannot overlap; asserted below

        out[:, j] = np.where(valid0, col, np.nan)
        out[n - 1, j] = np.nan            # last bar has no forward window

    lab = pd.DataFrame(out, index=close.index, columns=close.columns)
    s = lab.stack(future_stack=True)
    s.index.names = ["date", "ticker"]
    return s.rename("y")


def verify_exhaustive(close, up=0.06, down=0.03, horizon=5) -> dict:
    """Confirm the two win conditions never co-occur (the whole point).

    Recomputes both binary conditions independently and checks the overlap is
    empty. If this ever returns overlap > 0, the class definition is broken and
    nothing downstream can be trusted.
    """
    px = close.to_numpy(dtype=float)
    n, k = px.shape
    lw = np.zeros((n, k), bool)
    sw = np.zeros((n, k), bool)

    for j in range(k):
        p = px[:, j]
        a = np.full(n, np.inf); b = np.full(n, np.inf)
        c = np.full(n, np.inf); d = np.full(n, np.inf)
        for h in range(1, horizon + 1):
            fp = np.full(n, np.nan); fp[: n - h] = p[h:]
            r = fp / p - 1.0; ok = np.isfinite(r)
            a = np.where(ok & (r >= up) & (h < a), h, a)
            b = np.where(ok & (r <= -down) & (h < b), h, b)
            c = np.where(ok & (r <= -up) & (h < c), h, c)
            d = np.where(ok & (r >= down) & (h < d), h, d)
        lw[:, j] = a < b
        sw[:, j] = c < d

    overlap = int((lw & sw).sum())
    return {
        "long_win": int(lw.sum()),
        "short_win": int(sw.sum()),
        "overlap": overlap,
        "exclusive": overlap == 0,
    }


def class_distribution(labels: pd.Series) -> dict:
    y = labels.dropna()
    n = len(y)
    return {
        "n": n,
        "short_win_pct": round(100 * float((y == SHORT_WIN).mean()), 2),
        "neither_pct": round(100 * float((y == NEITHER).mean()), 2),
        "long_win_pct": round(100 * float((y == LONG_WIN).mean()), 2),
    }
