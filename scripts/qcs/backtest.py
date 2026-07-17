"""Walk-forward backtest engine."""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from .config import Config
from .cv import PurgedWalkForward, information_coefficient, ic_summary
from .features import build_features, cross_sectional_zscore, make_labels
from .portfolio import (
    carry_cost,
    performance_stats,
    rebalance_cost,
    target_weights,
    turnover,
)


def _fit(X: pd.DataFrame, y: pd.Series, cfg: Config) -> xgb.XGBRegressor:
    m = cfg.model
    model = xgb.XGBRegressor(
        n_estimators=m.n_estimators,
        max_depth=m.max_depth,
        learning_rate=m.learning_rate,
        subsample=m.subsample,
        colsample_bytree=m.colsample_bytree,
        min_child_weight=m.min_child_weight,
        reg_lambda=m.reg_lambda,
        random_state=m.random_state,
        objective="reg:squarederror",
        n_jobs=4,
        verbosity=0,
    )
    model.fit(X, y, sample_weight=_uniqueness_weights(y, cfg.label.horizon))
    return model


def _uniqueness_weights(y: pd.Series, horizon: int) -> np.ndarray:
    """Down-weight overlapping labels.

    A 5-day label observed on each of 5 consecutive days is essentially the same
    observation counted 5 times. Without this correction the model over-fits the
    most heavily overlapped (i.e. most redundant) regions of the sample.
    """
    return np.full(len(y), 1.0 / horizon)


def walk_forward_predict(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: Config,
) -> tuple[pd.Series, pd.DataFrame, list]:
    """Train on expanding windows, predict strictly out-of-sample."""
    dates = X.index.get_level_values("date").unique()
    splitter = PurgedWalkForward(
        n_splits=cfg.cv.n_splits,
        purge=cfg.cv.purge,
        embargo=cfg.cv.embargo,
        min_train=cfg.cv.min_train,
    )

    preds = []
    importances = []

    for train_dates, test_dates in splitter.split(dates):
        tr = X.index.get_level_values("date").isin(train_dates)
        te = X.index.get_level_values("date").isin(test_dates)

        Xtr, ytr = X[tr], y[tr]
        # XGBoost handles NaN natively via default-direction splits. Dropping
        # incomplete rows throws away data for no reason -- require only that
        # the LABEL exists and that most features are present.
        ok = ytr.notna() & (Xtr.notna().sum(axis=1) >= 0.6 * Xtr.shape[1])
        Xtr, ytr = Xtr[ok], ytr[ok]

        if len(Xtr) < 200:
            continue

        model = _fit(Xtr, ytr, cfg)

        Xte = X[te]
        Xte = Xte[Xte.notna().sum(axis=1) >= 0.6 * Xte.shape[1]]
        if len(Xte) == 0:
            continue

        p = pd.Series(model.predict(Xte), index=Xte.index, name="pred")
        preds.append(p)

        importances.append(
            pd.Series(model.feature_importances_, index=X.columns)
        )

    if not preds:
        raise RuntimeError("No folds produced predictions.")

    pred = pd.concat(preds).sort_index()
    imp = pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False)
    return pred, imp, list(splitter.split(dates))


def rolling_betas(close: pd.DataFrame, market: pd.Series, window: int = 120) -> pd.DataFrame:
    """Rolling market beta per name, used to size the book beta-neutral."""
    r = close.pct_change(fill_method=None)
    m = market.pct_change().reindex(r.index)
    cov = r.rolling(window).cov(m)
    var = m.rolling(window).var()
    return cov.div(var, axis=0)


def simulate(
    pred: pd.Series,
    close: pd.DataFrame,
    cfg: Config,
    betas: pd.DataFrame | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """Simulate the long/short book with full cost accounting.

    Timing convention: scores computed from data up to and including date t are
    traded at the CLOSE of date t, and earn returns from t+1 onward. No lookahead.
    """
    ret = close.pct_change(fill_method=None)
    pred_dates = pred.index.get_level_values("date").unique().sort_values()

    rebal_dates = pred_dates[:: cfg.portfolio.rebalance_every]

    w = pd.Series(0.0, index=close.columns)
    rows = []

    for i, dt in enumerate(rebal_dates):
        scores = pred.xs(dt, level="date")
        b = betas.loc[dt] if betas is not None and dt in betas.index else None
        w_new = target_weights(scores, w, cfg.portfolio, betas=b)
        w_new = w_new.reindex(close.columns).fillna(0.0)

        tc = rebalance_cost(w_new, w, cfg.costs)
        to = turnover(w_new, w)

        # Hold period: from this rebalance to the next.
        if i + 1 < len(rebal_dates):
            nxt = rebal_dates[i + 1]
        else:
            nxt = pred_dates[-1]

        window = ret.loc[(ret.index > dt) & (ret.index <= nxt)]
        if len(window) == 0:
            w = w_new
            continue

        # Gross P&L of the held book over the window.
        gross = float((window.fillna(0.0) * w_new).sum(axis=1).add(1.0).prod() - 1.0)
        cc = carry_cost(w_new, cfg.costs, days=len(window))

        rows.append(
            {
                "date": dt,
                "gross": gross,
                "tcost": tc,
                "carry": cc,
                "net": gross - tc - cc,
                "turnover": to,
                "n_days": len(window),
                "n_long": int((w_new > 0).sum()),
                "n_short": int((w_new < 0).sum()),
            }
        )

        w = w_new

    ledger = pd.DataFrame(rows).set_index("date")
    return ledger["net"], ledger


def run(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    factor_close: pd.DataFrame,
    cfg: Config,
    shuffle_labels: bool = False,
) -> dict:
    """Full pipeline. `shuffle_labels=True` runs the NULL TEST."""
    panel = build_features(close, volume, factor_close)
    X = cross_sectional_zscore(panel)
    y = make_labels(close, cfg.label.horizon, cfg.label.demean)

    idx = X.index.intersection(y.index)
    X, y = X.loc[idx].sort_index(), y.loc[idx].sort_index()

    if shuffle_labels:
        # Shuffle WITHIN each date. Destroys the cross-sectional signal while
        # preserving every marginal distribution. A correct backtest must now
        # report an IC of zero and a net return that is negative by exactly the
        # cost drag. If it does not, the pipeline is leaking.
        rng = np.random.default_rng(cfg.model.random_state)
        y = y.groupby(level="date").transform(
            lambda s: pd.Series(rng.permutation(s.to_numpy()), index=s.index)
        )

    pred, importance, folds = walk_forward_predict(X, y, cfg)

    ic = information_coefficient(pred, y)
    ics = ic_summary(ic, cfg.label.horizon)

    betas = rolling_betas(close, factor_close.iloc[:, 0])
    net, ledger = simulate(pred, close, cfg, betas=betas)

    ppy = 252.0 / cfg.portfolio.rebalance_every
    stats = performance_stats(net, cfg.costs, ppy)

    gross_ann = float(ledger["gross"].mean() * ppy)
    tc_ann = float(ledger["tcost"].mean() * ppy)
    carry_ann = float(ledger["carry"].mean() * ppy)

    return {
        "ic": ic,
        "ic_summary": ics,
        "importance": importance,
        "ledger": ledger,
        "net_returns": net,
        "stats": stats,
        "gross_ann": gross_ann,
        "tcost_ann": tc_ann,
        "carry_ann": carry_ann,
        "avg_turnover": float(ledger["turnover"].mean()),
        "folds": folds,
    }
