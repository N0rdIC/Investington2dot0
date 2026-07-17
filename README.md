# Barrier Signal · Paper Trading Desk

A deployable paper-trading system for the triple-barrier model we backtested.
It generates daily long/short signals, runs a **virtual** portfolio with
**realistic fills**, and shows everything on a dashboard. No real money moves.

## Architecture (why it's split)

```
┌─────────────────────┐     commits      ┌──────────────────┐
│  GitHub Actions      │  state + json    │  Vercel          │
│  (daily cron + train)│ ───────────────► │  (static dash)   │
│  heavy compute, free │                  │  reads json      │
└─────────────────────┘                  └──────────────────┘
```

Vercel is serverless — it can't run XGBoost on a schedule or hold state. So the
compute (fetch prices → model → signals → virtual fills) runs as a **GitHub
Actions cron job** that commits results to JSON, and **Vercel just serves the
dashboard** that reads that JSON. Both are free at this scale.

## What each piece does

| File | Role |
|---|---|
| `scripts/train.py` | Fits the two barrier models on all history. **Manual trigger.** |
| `scripts/daily.py` | Daily: prices → signals → virtual portfolio → dashboard JSON. |
| `scripts/signal_engine.py` | Shared feature+scoring code (no train/serve skew). |
| `scripts/portfolio.py` | Virtual portfolio with **realistic pessimistic fills**. |
| `.github/workflows/daily.yml` | Cron (weekdays after US close) + manual button. |
| `.github/workflows/retrain.yml` | Manual retrain (type "retrain" to confirm). |
| `web/public/index.html` | The dashboard (single file, terminal style). |
| `web/public/dashboard.json` | Written by the daily job; read by the dashboard. |

## The fill model (the honest part)

The backtest used close-to-close barrier fills, which flatter the result. This
paper-trader is deliberately pessimistic so you learn the truth:

- **Entry:** next day's **open + slippage** (you act on yesterday's close signal,
  so you can't get yesterday's price).
- **Profit exit (+6%):** barrier price **minus** a slippage haircut (you don't
  get the exact touch).
- **Stop exit (−3%):** barrier **minus extra slippage**, and if price **gapped
  through** the stop, filled at the (worse) open. This is the single biggest
  difference from the backtest — stops don't hold on gaps.
- Every trade pays **commission + half-spread** both sides, plus **borrow** on
  shorts.

All slippage/cost knobs live in `portfolio.py` → `Costs`. Defaults are IBKR-ish;
bump `commission` to `0.0008` for Saxo.

## Setup (15 minutes)

### 1. Put this folder in a GitHub repo

```bash
cd deploy
git init && git add -A && git commit -m "barrier paper desk"
# create a repo on github, then:
git remote add origin https://github.com/YOU/barrier-desk.git
git push -u origin main
```

The `qcs/` package must sit inside `scripts/` (it's already copied there).

### 2. Train the first model

In the repo's **Actions** tab → **retrain-model** → **Run workflow** → type
`retrain`. This fits the models and commits `model/*.json`. Takes ~5 min.

(Or run locally: `cd scripts && pip install -r requirements.txt && python train.py`.)

### 3. Run the first daily job

**Actions** → **daily-signals** → **Run workflow**. It creates the first
`state/portfolio.json` and `web/public/dashboard.json`. After this it runs itself
every weekday at 22:15 UTC.

### 4. Deploy the dashboard to Vercel

- Import the repo at [vercel.com/new](https://vercel.com/new).
- Set **Root Directory** to `web`.
- Framework preset: **Other**. Build command: none. Output dir: `public`.
- Deploy. Your dashboard is live and refreshes as the daily job commits.

Because the dashboard reads `dashboard.json` from the same deployment, every
daily commit triggers a Vercel redeploy with fresh data automatically.

## Daily flow

1. 22:15 UTC weekdays, GitHub Actions wakes up.
2. Downloads recent prices for the universe.
3. Loads the frozen models, scores the latest bar.
4. Opens/closes virtual positions with realistic fills.
5. Commits updated state + dashboard JSON → Vercel redeploys.

## Retraining

Manual by design — you decide when the live model changes, so what trades is
always a version you chose. Click **retrain-model**, confirm, done. The daily job
picks up the new model on its next run. Consider retraining every 1–3 months;
watch the per-year AUC decay we saw in the backtest (0.74 → 0.62 over a decade).

## Reading the dashboard

- **Equity** — virtual account value after realistic costs. The amber dashed
  line is your $100k starting point.
- **Today's signals** — names that cleared breakeven P* today. These are what a
  live trader would enter tomorrow at the open.
- **Open positions** — currently held, with days-in-trade.
- **Model ranking** — the raw P(long)/P(short) for the top names, so you can see
  conviction, not just the final picks.
- **Recent trades** — closed trades tagged by exit reason. Watch the **stop**
  tags: if their net % is regularly worse than −3%, that's gap slippage, and
  it's the real-world cost the backtest hid.

## Honest caveats

- **This validates fills, not the edge.** The backtest already measured the edge
  out-of-sample. This tells you whether that edge survives realistic execution —
  which is the one thing the backtest couldn't.
- **yfinance is delayed/unofficial.** Fine for paper trading, not for real
  execution. Signals are end-of-day, not intraday.
- **Survivorship bias persists** — the universe is today's names.
- **Paper ≠ live.** Real trading adds emotion, real spreads, partial fills, and
  borrow availability. Treat a good paper run as necessary, not sufficient.

Not investment advice. Educational paper-trading tool.
