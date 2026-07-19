"""Virtual paper-trading portfolio with REALISTIC fills.

State is a JSON blob (../state/portfolio.json) so it persists across daily runs
via git commits. No database needed.

FILL MODEL (the honest part)
----------------------------
The backtest used close-to-close barrier fills, which are optimistic: in reality
a +6% barrier is touched intraday and you don't get filled exactly there, and a
-3% stop routinely gaps THROUGH its level and fills worse. We model this:

  entry:  filled at next day's OPEN + slippage (you act on yesterday's close
          signal, so you cannot get yesterday's close)
  profit exit (+G): filled at the barrier price MINUS a small slippage haircut
          (you don't get the exact touch)
  stop exit (-L):   filled at the barrier price MINUS extra gap slippage, and if
          the day GAPPED through the stop, filled at that day's open (worse)
  time exit: filled at close on the horizon day

Every fill also pays commission + half-spread. All costs are explicit and
tunable so you can see how much they eat.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path

STATE = Path(__file__).resolve().parent.parent / "state" / "portfolio.json"


@dataclass
class Costs:
    commission: float = 0.0008      # per side (Saxo); set 0.0004 for IBKR
    half_spread: float = 0.0003     # per side
    entry_slip: float = 0.0005      # adverse slippage entering at open
    profit_slip: float = 0.0010     # you don't get the exact barrier touch
    stop_slip: float = 0.0025       # stops slip more, esp. on gaps
    borrow_daily: float = 0.03 / 252  # short financing per day


@dataclass
class Position:
    ticker: str
    side: str            # "long" | "short"
    entry_date: str
    entry_price: float
    up: float
    down: float
    horizon: int
    p_at_entry: float
    days_held: int = 0

    def barrier_prices(self):
        if self.side == "long":
            return self.entry_price * (1 + self.up), self.entry_price * (1 - self.down)
        else:
            return self.entry_price * (1 - self.up), self.entry_price * (1 + self.down)


def _default_state():
    return {
        "cash": 100000.0,
        "start_equity": 100000.0,
        "open_positions": [],
        "closed_trades": [],
        "equity_curve": [],       # [{date, equity}]
        "last_processed_date": None,
    }


def load_state():
    if STATE.exists():
        return json.loads(STATE.read_text())
    return _default_state()


def save_state(s):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(s, indent=2))


def _fill_entry(open_px, side, costs):
    """Entry price after adverse slippage + costs (as an effective price)."""
    slip = 1 + costs.entry_slip if side == "long" else 1 - costs.entry_slip
    return open_px * slip


def process_day(state, date, ohlc, new_signals, costs: Costs,
                up, down, horizon, capital_per_trade):
    """Advance the virtual portfolio by one trading day.

    ohlc: dict ticker -> {open, high, low, close} for THIS date.
    new_signals: list of {ticker, side, p} generated from the PREVIOUS close.
    Returns updated state. Ordering: first update/close existing positions using
    today's high/low/close, then open new positions at today's open.
    """
    still_open = []
    for pd_ in state["open_positions"]:
        pos = Position(**pd_)
        bar = ohlc.get(pos.ticker)
        if bar is None:
            still_open.append(pd_)          # no data today, carry forward
            continue

        pos.days_held += 1
        up_px, dn_px = pos.barrier_prices()
        exit_price = None
        reason = None

        hi, lo, cl, op = bar["high"], bar["low"], bar["close"], bar["open"]

        if pos.side == "long":
            hit_up = hi >= up_px
            hit_dn = lo <= dn_px
        else:
            hit_up = lo <= up_px            # profit barrier for short is a low
            hit_dn = hi >= dn_px

        # resolve which barrier (if both in one day, assume the STOP -> worst case)
        if hit_dn:
            # gap-through check: if the day opened already beyond the stop, fill
            # at the open (worse); else fill at the barrier minus stop slippage
            if pos.side == "long":
                gapped = op <= dn_px
                raw = op if gapped else dn_px
                exit_price = raw * (1 - costs.stop_slip)
            else:
                gapped = op >= dn_px
                raw = op if gapped else dn_px
                exit_price = raw * (1 + costs.stop_slip)
            reason = "stop"
        elif hit_up:
            if pos.side == "long":
                exit_price = up_px * (1 - costs.profit_slip)
            else:
                exit_price = up_px * (1 + costs.profit_slip)
            reason = "profit"
        elif pos.days_held >= pos.horizon:
            exit_price = cl                 # time barrier: close
            reason = "time"

        if exit_price is None:
            still_open.append({**asdict(pos)})
            continue

        # realised P&L per unit capital
        if pos.side == "long":
            gross = exit_price / pos.entry_price - 1
        else:
            gross = pos.entry_price / exit_price - 1  # short profits when price falls

        # costs: commission + spread both sides, borrow for shorts
        cost = 2 * (costs.commission + costs.half_spread)
        if pos.side == "short":
            cost += costs.borrow_daily * pos.days_held
        net = gross - cost

        pnl_cash = capital_per_trade * net
        state["cash"] += capital_per_trade + pnl_cash  # return capital + P&L
        state["closed_trades"].append({
            "ticker": pos.ticker, "side": pos.side,
            "entry_date": pos.entry_date, "exit_date": date,
            "entry_price": round(pos.entry_price, 4),
            "exit_price": round(exit_price, 4),
            "days_held": pos.days_held, "reason": reason,
            "gross_pct": round(gross * 100, 3),
            "net_pct": round(net * 100, 3),
            "pnl_cash": round(pnl_cash, 2),
            "p_at_entry": round(pos.p_at_entry, 3),
        })

    state["open_positions"] = still_open

    # --- open new positions at today's open ---
    open_tickers = {p["ticker"] for p in state["open_positions"]}
    for sig in new_signals:
        tkr = sig["ticker"]
        if tkr in open_tickers:
            continue                        # already hold it
        bar = ohlc.get(tkr)
        if bar is None or state["cash"] < capital_per_trade:
            continue
        entry = _fill_entry(bar["open"], sig["side"], costs)
        state["cash"] -= capital_per_trade  # reserve capital
        state["open_positions"].append(asdict(Position(
            ticker=tkr, side=sig["side"], entry_date=date,
            entry_price=entry, up=up, down=down, horizon=horizon,
            p_at_entry=sig["p"],
        )))
        open_tickers.add(tkr)

    # --- mark-to-market equity ---
    mtm = state["cash"]
    for pd_ in state["open_positions"]:
        bar = ohlc.get(pd_["ticker"])
        px = bar["close"] if bar else pd_["entry_price"]
        if pd_["side"] == "long":
            val = capital_per_trade * (px / pd_["entry_price"])
        else:
            val = capital_per_trade * (2 - px / pd_["entry_price"])
        mtm += val
    state["equity_curve"].append({"date": date, "equity": round(mtm, 2)})
    state["last_processed_date"] = date
    return state
