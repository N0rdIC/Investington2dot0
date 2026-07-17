"""Daily job. Run by GitHub Actions after US market close.

Steps:
  1. download recent prices for the universe (yfinance)
  2. load the frozen pretrained models
  3. score the latest bar -> today's long/short signals
  4. advance the virtual portfolio one day (realistic fills)
  5. write state + a compact dashboard JSON

Idempotent-ish: it processes the most recent complete trading day. If run twice
on the same day it detects last_processed_date and skips re-trading, only
refreshing the signal list.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from qcs.config import Config
from qcs.data import load_yahoo
from signal_engine import latest_scores, pick_signals, load_models
from portfolio import load_state, save_state, process_day, Costs

ROOT = Path(__file__).resolve().parent.parent
PUBLIC = ROOT / "web" / "public"
PUBLIC.mkdir(parents=True, exist_ok=True)

CAPITAL_PER_TRADE = 10000.0
N_SIDE = 5


def main():
    cfg = Config()
    models, meta = load_models()
    universe = meta["universe"]

    # only need ~2 years of history to build all features for the latest bar
    start = (pd.Timestamp.today() - pd.Timedelta(days=900)).strftime("%Y-%m-%d")
    px = load_yahoo(universe, start, cfg.end, cache="cache_daily")
    fx = load_yahoo(cfg.sector_etfs, start, cfg.end, cache="cache_daily")
    sx = load_yahoo(cfg.stress_tickers, start, cfg.end, cache="cache_daily")

    close, volume = px["close"], px["volume"]
    close = close.loc[:, close.notna().sum() > 200]
    volume = volume.reindex(columns=close.columns)

    scores = latest_scores(close, volume, fx["close"], sx["close"], models, meta)
    signals = pick_signals(scores, meta, n_side=N_SIDE)
    signal_date = scores["date"].iloc[0] if len(scores) else None

    state = load_state()

    # the trading day we process is the latest bar in `close`
    last_bar = str(close.index[-1].date())

    if state.get("last_processed_date") == last_bar:
        print(f"Already processed {last_bar}; refreshing signals only.")
    else:
        # build today's OHLC dict from the latest bar
        # (yfinance 'close' frame is adjusted close; for O/H/L we refetch raw)
        raw = _latest_ohlc(universe, start, cfg.end)
        ohlc = {t: raw[t] for t in raw if t in close.columns}
        state = process_day(
            state, last_bar, ohlc, signals, Costs(),
            up=meta["up"], down=meta["down"], horizon=meta["horizon"],
            capital_per_trade=CAPITAL_PER_TRADE,
        )
        save_state(state)
        print(f"Processed {last_bar}: {len(signals)} signals, "
              f"{len(state['open_positions'])} open positions.")

    _export_dashboard(state, signals, signal_date, meta, scores)


def _latest_ohlc(universe, start, end):
    """Fetch raw OHLC for the most recent bar (yfinance, not adjusted)."""
    import yfinance as yf
    raw = yf.download(universe, start=start, end=end, auto_adjust=False,
                      progress=False, group_by="ticker")
    out = {}
    for t in universe:
        try:
            sub = raw[t].dropna()
            if len(sub) == 0:
                continue
            last = sub.iloc[-1]
            out[t] = {"open": float(last["Open"]), "high": float(last["High"]),
                      "low": float(last["Low"]), "close": float(last["Close"])}
        except (KeyError, IndexError):
            continue
    return out


def _export_dashboard(state, signals, signal_date, meta, scores):
    """Compact JSON the Vercel dashboard reads."""
    eq = state["equity_curve"]
    start_eq = state["start_equity"]
    cur_eq = eq[-1]["equity"] if eq else start_eq
    closed = state["closed_trades"]

    wins = [t for t in closed if t["net_pct"] > 0]
    total_ret = cur_eq / start_eq - 1

    # simple running stats
    import statistics
    daily = []
    for i in range(1, len(eq)):
        daily.append(eq[i]["equity"] / eq[i - 1]["equity"] - 1)
    sharpe = None
    if len(daily) > 5 and statistics.pstdev(daily) > 0:
        sharpe = (statistics.mean(daily) / statistics.pstdev(daily)) * (252 ** 0.5)

    dash = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "model_trained_at": meta.get("trained_at"),
        "signal_date": signal_date,
        "p_star": meta["p_star"],
        "barrier": {"up": meta["up"], "down": meta["down"], "horizon": meta["horizon"]},
        "equity": {
            "current": round(cur_eq, 2),
            "start": start_eq,
            "total_return_pct": round(total_ret * 100, 2),
            "sharpe_est": round(sharpe, 2) if sharpe else None,
            "max_drawdown_pct": round(_max_dd(eq) * 100, 2),
        },
        "stats": {
            "n_closed": len(closed),
            "n_open": len(state["open_positions"]),
            "win_rate_pct": round(100 * len(wins) / len(closed), 1) if closed else None,
            "avg_win_pct": round(statistics.mean([t["net_pct"] for t in wins]), 2) if wins else None,
            "avg_loss_pct": round(statistics.mean([t["net_pct"] for t in closed if t["net_pct"] <= 0]), 2)
                            if any(t["net_pct"] <= 0 for t in closed) else None,
        },
        "todays_signals": signals,
        "top_scores": [
            {"ticker": t, "p_long": round(float(r["p_long"]), 3),
             "p_short": round(float(r["p_short"]), 3)}
            for t, r in scores.head(15).iterrows()
        ],
        "open_positions": state["open_positions"],
        "recent_trades": closed[-20:][::-1],
        "equity_curve": eq[-250:],
    }
    (PUBLIC / "dashboard.json").write_text(json.dumps(dash, indent=2))
    print(f"Dashboard exported: equity {cur_eq:.0f} "
          f"({total_ret * 100:+.1f}%), {len(signals)} signals.")


def _max_dd(eq):
    if not eq:
        return 0.0
    peak = eq[0]["equity"]
    mdd = 0.0
    for e in eq:
        peak = max(peak, e["equity"])
        mdd = min(mdd, e["equity"] / peak - 1)
    return mdd


if __name__ == "__main__":
    main()
