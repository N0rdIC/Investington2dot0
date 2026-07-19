"""Historical portfolio simulation.

Runs a full trading simulation over the training history using ONLY the
walk-forward out-of-sample predictions. This matters: an equity curve built from
in-sample predictions is fiction, because the model saw those outcomes during
fitting. Every prediction used here was made by a model that had not seen the
period it is trading.

Mechanics:
  - start with `capital`, split into at most `max_positions` equal slots
  - each day: first resolve exits (barrier touched, or the time barrier), then
    open new positions from the highest-expectancy signals that clear the gate
  - a name already held is never re-entered
  - costs charged on entry and exit; shorts additionally pay borrow

Exits use the same barrier logic as the labels, so simulated outcomes agree with
what the model was trained to predict. This is deliberately the OPTIMISTIC fill
model (fills exactly at the barrier); the live paper trader in portfolio.py
applies pessimistic fills with gap slippage. Comparing the two tells you how
much of the edge execution costs you.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _find_exit(prices: np.ndarray, i0: int, up: float, down: float,
               horizon: int, side: str):
    """Walk forward from index i0. Returns (exit_idx, exit_price, reason)."""
    p0 = prices[i0]
    n = len(prices)
    end = min(i0 + horizon, n - 1)
    for i in range(i0 + 1, end + 1):
        p = prices[i]
        if not np.isfinite(p):
            continue
        r = p / p0 - 1.0
        if side == "long":
            if r >= up:
                return i, p0 * (1 + up), "win"
            if r <= -down:
                return i, p0 * (1 - down), "stop"
        else:
            if r <= -up:
                return i, p0 * (1 - up), "win"
            if r >= down:
                return i, p0 * (1 + down), "stop"
    return end, prices[end], "timeout"


def simulate_history(preds: dict, close: pd.DataFrame, meta: dict,
                     capital: float = 100000.0, max_positions: int = 10,
                     borrow_annual: float = 0.03):
    """preds: {"long": DataFrame, "short": DataFrame} indexed by (date, ticker)
    with an 'E' column (predicted expectancy, as a fraction)."""
    up, down = meta["up"], meta["down"]
    horizon = meta["horizon"]
    cost = meta.get("cost", 0.0014)
    gates = {s: meta.get(f"min_E_{s}") for s in ("long", "short")}

    # union of dates that have predictions
    all_dates = sorted(set().union(*[
        set(p.index.get_level_values("date").unique()) for p in preds.values()
    ]))
    if not all_dates:
        return None

    px_index = {d: i for i, d in enumerate(close.index)}
    arrays = {t: close[t].to_numpy(dtype=float) for t in close.columns}

    slot = capital / max_positions
    cash = capital
    open_pos = []          # dicts
    trades = []
    curve = []

    for dt in all_dates:
        i = px_index.get(dt)
        if i is None:
            continue

        # ---- resolve exits ----
        still = []
        for pos in open_pos:
            if i >= pos["exit_idx"]:
                gross = ((pos["exit_price"] / pos["entry_price"] - 1.0)
                         if pos["side"] == "long"
                         else (pos["entry_price"] / pos["exit_price"] - 1.0))
                days = pos["exit_idx"] - pos["entry_idx"]
                c = cost + (borrow_annual * days / 252.0 if pos["side"] == "short" else 0.0)
                net = gross - c
                cash += slot * (1.0 + net)
                trades.append({
                    "ticker": pos["ticker"], "side": pos["side"],
                    "size": round(slot, 2),
                    "shares": round(slot / pos["entry_price"], 2),
                    "entry_date": str(pd.Timestamp(pos["entry_date"]).date()),
                    "exit_date": str(pd.Timestamp(close.index[pos["exit_idx"]]).date()),
                    "entry_price": round(pos["entry_price"], 3),
                    "exit_price": round(pos["exit_price"], 3),
                    "days": int(days), "reason": pos["reason"],
                    "gross_pct": round(gross * 100, 3),
                    "net_pct": round(net * 100, 3),
                    "pnl": round(slot * net, 2),
                    "E_at_entry_pct": round(pos["E"] * 100, 3),
                })
            else:
                still.append(pos)
        open_pos = still

        # ---- open new positions ----
        held = {p["ticker"] for p in open_pos}
        free = max_positions - len(open_pos)
        if free > 0:
            cands = []
            for side, P in preds.items():
                g = gates.get(side)
                if g is None:
                    continue
                try:
                    day = P.xs(dt, level="date")
                except KeyError:
                    continue
                sub = day[day["E"] >= g]
                for tkr, row in sub.iterrows():
                    if tkr in held or tkr not in arrays:
                        continue
                    cands.append((float(row["E"]), tkr, side))
            cands.sort(reverse=True)

            for E, tkr, side in cands[:free]:
                if cash < slot:
                    break
                arr = arrays[tkr]
                p0 = arr[i]
                if not np.isfinite(p0) or p0 <= 0:
                    continue
                ei, ep, reason = _find_exit(arr, i, up, down, horizon, side)
                if ei <= i:
                    continue
                cash -= slot
                open_pos.append({
                    "ticker": tkr, "side": side, "entry_idx": i, "entry_date": dt,
                    "entry_price": p0, "exit_idx": ei, "exit_price": ep,
                    "reason": reason, "E": E,
                })
                held.add(tkr)

        # ---- mark to market ----
        mtm = cash
        for pos in open_pos:
            p = arrays[pos["ticker"]][i]
            if not np.isfinite(p):
                p = pos["entry_price"]
            r = ((p / pos["entry_price"] - 1.0) if pos["side"] == "long"
                 else (pos["entry_price"] / p - 1.0))
            mtm += slot * (1.0 + r)
        curve.append({"date": str(pd.Timestamp(dt).date()), "equity": round(mtm, 2)})

    return {"equity_curve": curve, "trades": trades,
            "stats": {**_stats(curve, trades, capital),
                      "position_size": round(capital / max_positions, 2),
                      "max_positions": max_positions}}


def _stats(curve, trades, capital):
    if not curve:
        return {}
    eq = np.array([c["equity"] for c in curve], dtype=float)
    rets = np.diff(eq) / eq[:-1]
    rets = rets[np.isfinite(rets)]

    years = max(1e-9, len(curve) / 252.0)
    total = eq[-1] / capital - 1.0
    cagr = (eq[-1] / capital) ** (1 / years) - 1.0 if eq[-1] > 0 else -1.0
    vol = float(rets.std() * np.sqrt(252)) if len(rets) > 2 else 0.0
    sharpe = (float(rets.mean()) * 252 / vol) if vol > 0 else None
    peak = np.maximum.accumulate(eq)
    dd = float((eq / peak - 1.0).min())

    wins = [t for t in trades if t["net_pct"] > 0]
    losses = [t for t in trades if t["net_pct"] <= 0]
    gp = sum(t["pnl"] for t in wins)
    gl = -sum(t["pnl"] for t in losses)

    by_reason = {}
    for r in ("win", "stop", "timeout"):
        sub = [t for t in trades if t["reason"] == r]
        if sub:
            by_reason[r] = {
                "n": len(sub),
                "pct": round(100 * len(sub) / len(trades), 1),
                "avg_net_pct": round(float(np.mean([t["net_pct"] for t in sub])), 3),
            }
    by_side = {}
    for s in ("long", "short"):
        sub = [t for t in trades if t["side"] == s]
        if sub:
            w = [t for t in sub if t["net_pct"] > 0]
            by_side[s] = {
                "n": len(sub),
                "win_rate_pct": round(100 * len(w) / len(sub), 1),
                "avg_net_pct": round(float(np.mean([t["net_pct"] for t in sub])), 3),
                "total_pnl": round(sum(t["pnl"] for t in sub), 2),
            }

    return {
        "start_capital": capital,
        "position_size": round(capital / 10.0, 2),
        "n_wins": len(wins),
        "n_losses": len(losses),
        "gross_profit": round(gp, 2),
        "gross_loss": round(gl, 2),
        "net_pnl": round(gp - gl, 2),
        "best_trade_pct": round(max((t["net_pct"] for t in trades), default=0), 2),
        "worst_trade_pct": round(min((t["net_pct"] for t in trades), default=0), 2),
        "avg_days_held": round(float(np.mean([t["days"] for t in trades])), 1) if trades else None,
        "final_equity": round(float(eq[-1]), 2),
        "total_return_pct": round(total * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "vol_pct": round(vol * 100, 2),
        "sharpe": round(sharpe, 2) if sharpe else None,
        "max_drawdown_pct": round(dd * 100, 2),
        "n_trades": len(trades),
        "trades_per_year": round(len(trades) / years, 1),
        "win_rate_pct": round(100 * len(wins) / len(trades), 2) if trades else None,
        "avg_win_pct": round(float(np.mean([t["net_pct"] for t in wins])), 3) if wins else None,
        "avg_loss_pct": round(float(np.mean([t["net_pct"] for t in losses])), 3) if losses else None,
        "expectancy_pct": round(float(np.mean([t["net_pct"] for t in trades])), 3) if trades else None,
        "profit_factor": round(gp / gl, 2) if gl > 0 else None,
        "by_reason": by_reason,
        "by_side": by_side,
        "period_start": curve[0]["date"],
        "period_end": curve[-1]["date"],
        "years": round(years, 1),
    }
