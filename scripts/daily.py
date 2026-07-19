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

    # attach indicative barrier levels to each signal, based on the last close.
    # NOTE: these are INDICATIVE. The real entry is tomorrow's open, so actual
    # target/stop will be set from that fill, not from this reference price.
    for s in signals:
        try:
            ref = float(close[s["ticker"]].dropna().iloc[-1])
        except (KeyError, IndexError):
            continue
        s["ref_price"] = round(ref, 2)
        if s["side"] == "long":
            s["target_price"] = round(ref * (1 + meta["up"]), 2)
            s["stop_price"] = round(ref * (1 - meta["down"]), 2)
        else:
            s["target_price"] = round(ref * (1 - meta["up"]), 2)
            s["stop_price"] = round(ref * (1 + meta["down"]), 2)

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


def _realized_glp(closed, meta, costs_rt=0.0014):
    """Realised G / L / P from actual closed trades.

    This is the loop closing back on the original framework:
        E = P*G - (1-P)*L - c
    The model ASSUMED G=+6%, L=-3%. What did we actually GET? Stops that gap
    through fill worse than -3%, and profit exits fill slightly under +6%, so
    realised G and L drift from target. Comparing realised P against the
    breakeven P* implied by realised G/L is the honest scorecard.
    """
    if not closed:
        return None

    wins = [t["net_pct"] / 100 for t in closed if t["net_pct"] > 0]
    losses = [t["net_pct"] / 100 for t in closed if t["net_pct"] <= 0]
    n = len(closed)

    P = len(wins) / n
    G = sum(wins) / len(wins) if wins else 0.0
    L = abs(sum(losses) / len(losses)) if losses else 0.0

    # breakeven win rate implied by the REALISED payoff geometry
    p_star_real = (L + costs_rt) / (G + L) if (G + L) > 0 else None
    expectancy = P * G - (1 - P) * L

    # split by exit reason - where the money actually comes from
    by_reason = {}
    for r in ("profit", "stop", "time"):
        sub = [t for t in closed if t["reason"] == r]
        if sub:
            by_reason[r] = {
                "n": len(sub),
                "pct_of_trades": round(100 * len(sub) / n, 1),
                "avg_net_pct": round(sum(t["net_pct"] for t in sub) / len(sub), 2),
            }

    # long vs short breakdown
    by_side = {}
    for s in ("long", "short"):
        sub = [t for t in closed if t["side"] == s]
        if sub:
            w = [t for t in sub if t["net_pct"] > 0]
            by_side[s] = {
                "n": len(sub),
                "win_rate_pct": round(100 * len(w) / len(sub), 1),
                "avg_net_pct": round(sum(t["net_pct"] for t in sub) / len(sub), 2),
            }

    return {
        "P_realized_pct": round(P * 100, 1),
        "G_realized_pct": round(G * 100, 2),
        "L_realized_pct": round(L * 100, 2),
        "G_target_pct": round(meta["up"] * 100, 1),
        "L_target_pct": round(meta["down"] * 100, 1),
        "breakeven_P_pct": round(p_star_real * 100, 1) if p_star_real else None,
        "expectancy_pct": round(expectancy * 100, 3),
        "edge_vs_breakeven_pct": round((P - p_star_real) * 100, 1) if p_star_real else None,
        "n_trades": n,
        "by_reason": by_reason,
        "by_side": by_side,
    }


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

    # enrich open positions with their barrier target / stop prices
    open_enriched = []
    for p in state["open_positions"]:
        ep = p["entry_price"]
        if p["side"] == "long":
            tgt, stp = ep * (1 + p["up"]), ep * (1 - p["down"])
        else:
            tgt, stp = ep * (1 - p["up"]), ep * (1 + p["down"])
        open_enriched.append({
            **p,
            "target_price": round(tgt, 2),
            "stop_price": round(stp, 2),
            "days_left": max(0, p["horizon"] - p["days_held"]),
        })

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
        "glp": _realized_glp(closed, meta),
        "todays_signals": signals,
        "top_longs": [
            {"ticker": t, "p": round(float(r["p_win_long"]), 3),
             "p_stop": round(float(r["p_stop_long"]), 3),
             "E_pct": round(float(r["E_long"]) * 100, 3)}
            for t, r in scores.nlargest(10, "E_long").iterrows()
        ],
        "top_shorts": [
            {"ticker": t, "p": round(float(r["p_win_short"]), 3),
             "p_stop": round(float(r["p_stop_short"]), 3),
             "E_pct": round(float(r["E_short"]) * 100, 3)}
            for t, r in scores.nlargest(10, "E_short").iterrows()
        ],
        "gate_long_pct": (meta.get("min_E_long") or 0) * 100,
        "gate_short_pct": (meta.get("min_E_short") or 0) * 100,
        "horizon": meta.get("horizon"),
        "model_type": meta.get("model_type", "2model"),
        "training_history": _load_history(),
        "calibration": _load_calibration_curve(),
        "sweep": _load_sweep(),
        "open_positions": open_enriched,
        "recent_trades": closed[-25:][::-1],
        "equity_curve": eq[-250:],
    }
    (PUBLIC / "dashboard.json").write_text(json.dumps(dash, indent=2))
    print(f"Dashboard exported: equity {cur_eq:.0f} "
          f"({total_ret * 100:+.1f}%), {len(signals)} signals.")


def _load_sweep():
    """The precision-vs-volume tradeoff from the last training."""
    h = ROOT / "model" / "training_history.json"
    if not h.exists():
        return None
    try:
        hist = json.loads(h.read_text())
    except json.JSONDecodeError:
        return None
    if not hist:
        return None
    m = hist[-1].get("metrics", {})
    return {"sweep": m.get("sweep_long", []),
            "sweep_short": m.get("sweep_short", []),
            "recommended": m.get("recommended_long"),
            "recommended_short": m.get("recommended_short"),
            "precision_by_year": m.get("precision_by_year_long", {}),
            "p_star_pct": m.get("p_star_pct"),
            "base_rate_pct": m.get("base_rate_long_pct"),
            "auc": m.get("auc_long"),
            "auc_short": m.get("auc_short")}


def _load_calibration_curve():
    """The measured map from softmax score -> actual win rate."""
    c = ROOT / "model" / "calibration.json"
    if not c.exists():
        return None
    try:
        d = json.loads(c.read_text())
    except json.JSONDecodeError:
        return None
    return {"curve_long": d.get("curve_long", []),
            "metrics_long": d.get("metrics_long", {}),
            "metrics_short": d.get("metrics_short", {})}


def _load_history():
    """Training metrics from every retrain, so the dashboard can show how the
    model's EXPECTED success rate has evolved -- and compare it to the REALISED
    rate from paper trading."""
    h = ROOT / "model" / "training_history.json"
    if not h.exists():
        return []
    try:
        hist = json.loads(h.read_text())
    except json.JSONDecodeError:
        return []
    out = []
    for e in hist[-12:]:
        m = e.get("metrics", {})
        out.append({
            "trained_at": e.get("trained_at"),
            "train_end": e.get("train_end"),
            "n_samples": e.get("n_samples"),
            "n_names": e.get("n_names"),
            "auc_long": m.get("auc_long"),
            "auc_short": m.get("auc_short"),
            "expected_P_long_pct": m.get("expected_P_long_pct"),
            "expected_P_short_pct": m.get("expected_P_short_pct"),
            "p_star_pct": m.get("p_star_pct"),
            "edge_vs_pstar_pp": m.get("edge_vs_pstar_pp"),
            "expected_E_per_trade_pct": m.get("expected_E_per_trade_pct"),
            "signal_rate_pct": m.get("signal_rate_pct"),
            "base_rate_long_pct": m.get("base_rate_long_pct"),
        })
    return out


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
