"""Configuration for the cross-sectional ranker backtest."""
from dataclasses import dataclass, field


# ~90-name CROSS-SECTOR high-vol universe. The point of the breadth is not
# more names in the same cluster (that does nothing for effective N) but names
# with DIFFERENT drivers, so the effective number of independent bets rises.
# Re-screen monthly; membership in the high-vol bucket churns fast.
UNIVERSE = [
    # --- crypto-proxy ---
    "MSTR", "COIN", "MARA", "RIOT", "HOOD", "CLSK", "BITF", "CIFR",
    # --- AI / semis ---
    "NVDA", "AMD", "MU", "SMCI", "AVGO", "MRVL", "QCOM", "LRCX", "AMAT", "TSM",
    # --- AI infra / quantum ---
    "IONQ", "RGTI", "SOUN", "AI", "QBTS",
    # --- high-multiple software ---
    "PLTR", "APP", "CRWD", "NET", "SNOW", "DDOG", "ZS", "PANW", "MDB", "TEAM",
    # --- consumer / fintech ---
    "TSLA", "CVNA", "AFRM", "SOFI", "UPST", "DKNG", "RIVN", "LCID", "ABNB", "SQ",
    # --- biotech (different driver: trials, FDA) ---
    "MRNA", "BNTX", "CRSP", "NTLA", "BEAM", "VKTX", "SRPT", "EXAS", "ALNY",
    # --- energy / commodities (different driver: oil, gas) ---
    "OXY", "DVN", "FANG", "MRO", "APA", "HAL", "SLB", "FCX", "AA",
    # --- financials / high-beta cyclicals ---
    "SCHW", "COF", "SYF", "ALLY", "KEY", "RF",
    # --- industrials / EV / clean ---
    "ENPH", "FSLR", "PLUG", "CHPT", "RUN", "BE",
    # --- China ADRs (different driver: China macro) ---
    "BABA", "PDD", "JD", "BIDU", "NIO", "XPEV", "LI",
    # --- misc high-vol / meme / travel ---
    "GME", "AMC", "CCL", "NCLH", "UAL", "AAL", "RBLX", "U", "DASH", "SHOP",
    # --- space / new industrials ---
    "RKLB", "ASTS", "LUNR",
    # --- LONG-LIVED high-vol names with REAL 2008-2009 (GFC) data ---
    # These existed through the crisis, so extending the start date to ~2005
    # gives the model genuine crisis-regime observations instead of NaN.
    "WYNN", "LVS", "MGM",           # casinos - violently cyclical, pre-2008
    "X", "CLF", "AA",               # steel / materials - deep cyclicals
    "F", "GM",                      # autos (GM re-IPO'd 2010; F has full history)
    "BAC", "C", "WFC",              # money-center banks - GFC epicenter
    "GS", "MS",                     # investment banks
    "DAL", "LUV",                   # airlines
    "HAL", "SLB", "OXY", "DVN",     # energy (already some above; dedup handled)
    "NEM", "GOLD",                  # gold miners - crisis hedge behaviour
    "WDC", "STX",                   # disk drives - high-beta tech cyclical
    "AMAT", "KLAC", "MU", "NVDA",   # semis with full history
    "CAT", "DE",                    # heavy industrials
    "URBN", "GES",                  # high-beta retail
]

# Names known to have deep history (>= 2005). Used only for documentation /
# optional filtering; the loader keeps whatever Yahoo returns.
LONG_LIVED = [
    "WYNN", "LVS", "MGM", "X", "CLF", "AA", "F", "BAC", "C", "WFC", "GS", "MS",
    "DAL", "LUV", "HAL", "SLB", "OXY", "DVN", "NEM", "GOLD", "WDC", "STX",
    "AMAT", "KLAC", "MU", "NVDA", "CAT", "DE", "URBN", "GES", "AMD", "AVGO",
]

# Sector proxies used to strip out systematic moves and isolate the residual.
SECTOR_ETFS = ["SPY", "SMH", "XLK", "ARKK"]

# Regime / stress proxies. VIXY doesn't reach 2008, so we derive a synthetic
# VIX-like stress measure from SPY realized vol in features instead; these ETFs
# are additional cross-sectional factors and regime context where available.
STRESS_TICKERS = ["SPY", "TLT", "GLD", "HYG", "UUP", "XLF", "XLE"]


@dataclass
class Costs:
    """All costs as decimal fractions of notional."""

    # Round-trip cost: commission both legs + spread + slippage.
    # Saxo France, USD sub-account: 0.08% x 2 + ~2bp spread.
    round_trip: float = 0.0020

    # Annualised stock-borrow / financing cost applied to SHORT notional only.
    borrow_annual: float = 0.03

    # Flat tax (France, CTO) applied to net annual realised gains.
    flat_tax: float = 0.30

    @property
    def one_way(self) -> float:
        return self.round_trip / 2.0


@dataclass
class PortfolioCfg:
    n_long: int = 5
    n_short: int = 5

    # Hysteresis buffer: an existing long is only closed once its rank falls
    # outside the top `buffer_rank`. Set == n_long to disable. This is the
    # single most effective turnover reducer available.
    buffer_rank: int = 10

    # Gross exposure as a fraction of capital. 1.0 => 50% long + 50% short.
    gross: float = 1.0

    rebalance_every: int = 5   # trading days between rebalances
    allow_short: bool = True


@dataclass
class LabelCfg:
    horizon: int = 10          # forward return horizon in trading days
    demean: bool = True        # cross-sectionally demean -> pure ranking target


@dataclass
class CVCfg:
    n_splits: int = 6
    # Purge = label horizon. Any train sample whose label window overlaps the
    # test window is dropped. Without this, the backtest leaks and lies.
    purge: int = 10            # must equal the label horizon
    # Embargo: extra days dropped after the test block to kill serial-correlation
    # leakage from features (e.g. a 60-day EWMA straddling the boundary).
    embargo: int = 10
    min_train: int = 500       # trading days


@dataclass
class ModelCfg:
    n_estimators: int = 300
    max_depth: int = 3         # shallow: SNR here is ~1-5%, deep trees memorise noise
    learning_rate: float = 0.03
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: float = 20.0
    reg_lambda: float = 5.0
    random_state: int = 0


@dataclass
class Config:
    universe: list = field(default_factory=lambda: list(UNIVERSE))
    sector_etfs: list = field(default_factory=lambda: list(SECTOR_ETFS))
    stress_tickers: list = field(default_factory=lambda: list(STRESS_TICKERS))
    costs: Costs = field(default_factory=Costs)
    portfolio: PortfolioCfg = field(default_factory=PortfolioCfg)
    label: LabelCfg = field(default_factory=LabelCfg)
    cv: CVCfg = field(default_factory=CVCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    # Column-sampling weight for volatility features (< 1 caps their dominance).
    model_vol_cap_weight: float = 0.25
    # Minimum predicted expectancy (as a fraction) required to fire a signal.
    # 0.005 = 0.5%. A margin above zero absorbs estimation error in p_win/p_stop,
    # gap slippage on stops, and borrow cost on the short book.
    min_expectancy: float = 0.005
    start: str = "2005-01-01"
    end: str = "2026-07-01"
