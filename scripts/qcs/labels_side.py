"""Per-side outcome labels: what actually happens to THIS trade.

WHY THIS REPLACES THE JOINT LABELS
----------------------------------
The joint labelling {SHORT_WIN, NEITHER, LONG_WIN} answers "which side wins".
That is not the same question as "what happens to my long trade", and the
difference matters enormously for expectancy.

Measured on real paths, the joint class NEITHER decomposes as:

    ~60% -> the long trade STOPPED OUT at -L  (a real loss)
    ~40% -> the long trade TIMED OUT          (exits near flat)

So NEITHER is a mixture, and p_stop for a long trade is smeared across two
joint classes in a proportion the model never predicts. Without p_stop you
cannot compute expectancy, and the naive substitute

    P* = (L + c) / (G + L)

silently assumes every non-win is a full -L loss. With a time barrier that is
false, and it makes a profitable model look unprofitable.

THE CORRECT OUTCOME SPACE (per side)
------------------------------------
    class 2  WIN      : the profit barrier is touched first
    class 1  TIMEOUT  : neither barrier touched within the horizon
    class 0  STOP     : the stop barrier is touched first

Mutually exclusive, exhaustive, and defined FOR A SPECIFIC TRADE. One softmax
per side then yields p_win + p_timeout + p_stop = 1, and expectancy is direct:

    E = p_win*G - p_stop*L + p_timeout*r_timeout - c

For a long, the barriers are (+up, -down); for a short they mirror to
(-up, +down) and the sign of the realised return flips.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STOP, TIMEOUT, WIN = 0, 1, 2


def side_outcome_labels(
    close: pd.DataFrame,
    up: float = 0.06,
    down: float = 0.03,
    horizon: int = 10,
    side: str = "long",
) -> pd.Series:
    """0 = STOP, 1 = TIMEOUT, 2 = WIN for the given trade direction."""
    if side not in ("long", "short"):
        raise ValueError("side must be 'long' or 'short'")

    px = close.to_numpy(dtype=float)
    n, k = px.shape
    out = np.full((n, k), np.nan)

    for j in range(k):
        p = px[:, j]
        valid = np.isfinite(p) & (p > 0)
        d_win = np.full(n, np.inf)
        d_stop = np.full(n, np.inf)

        for h in range(1, horizon + 1):
            fp = np.full(n, np.nan)
            fp[: n - h] = p[h:]
            r = fp / p - 1.0
            ok = np.isfinite(r)
            if side == "long":
                hit_w, hit_s = (r >= up), (r <= -down)
            else:
                hit_w, hit_s = (r <= -up), (r >= down)
            d_win = np.where(ok & hit_w & (h < d_win), h, d_win)
            d_stop = np.where(ok & hit_s & (h < d_stop), h, d_stop)

        col = np.full(n, float(TIMEOUT))
        col[d_win < d_stop] = WIN
        col[d_stop < d_win] = STOP
        out[:, j] = np.where(valid, col, np.nan)
        out[n - 1, j] = np.nan

    lab = pd.DataFrame(out, index=close.index, columns=close.columns)
    s = lab.stack(future_stack=True)
    s.index.names = ["date", "ticker"]
    return s.rename("y")


def timeout_returns(close, up=0.06, down=0.03, horizon=10, side="long") -> pd.Series:
    """Realised return of the trades that TIME OUT.

    r_timeout is usually small but not exactly zero, and it enters expectancy
    with a large weight when the time barrier binds. Measure it, don't assume it.
    """
    px = close.to_numpy(dtype=float)
    n, k = px.shape
    out = np.full((n, k), np.nan)

    for j in range(k):
        p = px[:, j]
        d_win = np.full(n, np.inf)
        d_stop = np.full(n, np.inf)
        for h in range(1, horizon + 1):
            fp = np.full(n, np.nan); fp[: n - h] = p[h:]
            r = fp / p - 1.0; ok = np.isfinite(r)
            if side == "long":
                hw, hs = (r >= up), (r <= -down)
            else:
                hw, hs = (r <= -up), (r >= down)
            d_win = np.where(ok & hw & (h < d_win), h, d_win)
            d_stop = np.where(ok & hs & (h < d_stop), h, d_stop)

        timed = (~np.isfinite(d_win)) & (~np.isfinite(d_stop))
        fp = np.full(n, np.nan); fp[: n - horizon] = p[horizon:]
        r_end = fp / p - 1.0
        if side == "short":
            r_end = -r_end
        out[:, j] = np.where(timed, r_end, np.nan)

    s = pd.DataFrame(out, index=close.index, columns=close.columns).stack(future_stack=True)
    s.index.names = ["date", "ticker"]
    return s.rename("r_timeout")


def outcome_distribution(labels: pd.Series) -> dict:
    y = labels.dropna()
    return {
        "n": int(len(y)),
        "stop_pct": round(100 * float((y == STOP).mean()), 2),
        "timeout_pct": round(100 * float((y == TIMEOUT).mean()), 2),
        "win_pct": round(100 * float((y == WIN).mean()), 2),
    }


def true_breakeven(p_stop: float, G: float, L: float, c: float) -> float:
    """Win rate needed to break even GIVEN the stop rate.

        E = p_win*G - p_stop*L - c = 0  =>  p_win = (p_stop*L + c) / G

    Contrast with the naive P* = (L+c)/(G+L), which assumes p_timeout = 0 and
    is therefore too strict whenever the time barrier binds.
    """
    return (p_stop * L + c) / G
