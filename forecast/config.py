"""Configuration shared by training and the daily run (values taken from the team's lasso.ipynb)."""
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "Data" / "raw"
MODELS_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "Outputs"


@dataclass
class Config:
    # ---- data ----
    target: str = "brent_fut"
    # quoted as levels / rates / indices: first differences plus the level
    level_series: List[str] = field(default_factory=lambda: [
        "vix", "ovx", "ust10y", "curve_10y2y", "t10y3m", "breakeven10y", "epu"])
    exclude: List[str] = field(default_factory=lambda: ["wti_spot", "brent_spot"])   # published with delays
    ffill_limit: int = 3
    slow_ffill_limit: int = 7
    feature_lag: int = 0
    fred_lag: int = 1                 # FRED daily series known the next day
    season_years: int = 5
    weekly_stale_days: int = 21
    weekly_release_lag: int = 6       # weekly series enter 6 days after the period date

    # series the LASSO-family and MS-AR plugins forecast (each gets its own panel in the e-mail)
    targets: List[str] = field(default_factory=lambda: ["brent_fut", "wti_fut"])
    # a target's prices before this date are not used: WTI settled at -37.63 on 2020-04-20, which would put a -60% "return"
    # into the estimation windows. (Other series, e.g. WTI as a predictor of Brent, are untouched.)
    series_start: dict = field(default_factory=lambda: {"wti_fut": "2020-06-01"})

    # ---- forecasting ----
    horizons: List[int] = field(default_factory=lambda: [1, 5, 10, 20, 40, 60])
    window: int = 1260                # rolling estimation window (trading days)
    cv_folds: int = 5
    n_alphas: int = 40
    lasso_eps: float = 1e-2
    fit_intercept: bool = False
    pca_kmax: int = 8
    pca_k: Optional[int] = None
    tpca_t: float = 1.65
    rs_patterns: tuple = ("__lvl", "__anom", "20")
    seed: int = 0

    # ---- regimes ----
    hmm_k: int = 2
    hmm_inputs: List[str] = field(default_factory=lambda: ["ret"])
    hmm_min_obs: int = 500
    hmm_refit: int = 63
    hmm_window: Optional[int] = None

    # ---- ARMA-GARCH plugin (teammate's gp_arma_garch.ipynb: MA(2) mean, GARCH(1,1)-t on its residuals) ----
    arma_order: tuple = (0, 2)                        # (p, q); zero-mean ARMA fitted on daily log returns
    garch_pq: tuple = (1, 1)
    garch_dist: str = "t"
    arma_garch_targets: List[str] = field(default_factory=lambda: ["brent_fut", "wti_fut"])

    # ---- MS-AR plugin (teammate's msar.ipynb: 2-regime Markov-switching AR on Brent returns) ----
    # model name -> family ("baseline" = V1 MS-AR; "msarx" = V2 MS-ARX with lagged WTI/DXY/spread inputs),
    # estimation window (returns), and the mean weight that scales the expected return (0 = no change, 1 = full).
    # Values are the ones the notebook selected on its 2022-2023 validation sample.
    msar_models: dict = field(default_factory=lambda: {
        "MS-AR": {"family": "baseline", "window": 1260, "weight": 1.0},
        "MS-ARX": {"family": "msarx", "window": 756, "weight": 0.25},
    })
    msar_refit_every: int = 63            # trading days between re-estimations (about a quarter)
    msar_seed: int = 5210
    msar_n_sim: int = 4000                # simulated paths for the +-1 sd band
    msar_fill_limit: int = 3              # causal forward-fill limit for WTI / DXY quotes

    # ---- frozen-model evaluation (scripts/frozen_eval.py) ----
    frozen_cutoff: str = "2024-12-31"        # models are estimated / selected on data up to and including this date
    frozen_fan_origin: str = "2026-07-01"    # origin of the one-off "forecast fan vs actual" picture
    frozen_horizons: tuple = (1, 2, 3, 5, 10, 20, 40, 60)   # evaluated horizons (tables); charts use the short ones
    frozen_chart_horizon: int = 3            # rolling h-day-ahead chart
    frozen_trade_horizon: int = 3            # horizon of the trading rule shown in the e-mail table and equity chart
    frozen_zoom_days: int = 90               # price panel of the rolling chart shows the last N origins
    signal_horizon: int = 1                  # horizon of the BUY / SELL / HOLD summary at the top of the e-mail
    cost_bps: float = 5.0                    # transaction cost per unit of turnover (basis points)
    trade_band: float = 0.1                  # trade only if |forecast| > band * (1-sd of the h-day return)

    # ---- combination ----
    combo_pool: List[str] = field(default_factory=lambda: ["RW", "LASSO", "LASSO_1se", "PCA_reg", "TPCA", "LASSO_RS"])

    def to_dict(self):
        return asdict(self)
