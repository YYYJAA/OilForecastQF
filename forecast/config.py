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

    # ---- combination ----
    combo_pool: List[str] = field(default_factory=lambda: ["RW", "LASSO", "LASSO_1se", "PCA_reg", "TPCA", "LASSO_RS"])

    def to_dict(self):
        return asdict(self)
