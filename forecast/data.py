"""Load the raw CSVs written by scripts/update_data.py. No downloading happens here."""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from .config import RAW_DIR, Config

FRED_DAILY = ["ust10y", "curve_10y2y", "t10y3m", "breakeven10y", "ovx", "epu"]
WEEKLY = ["crude_stocks", "wei"]


@dataclass
class Inputs:
    prices: pd.DataFrame                      # daily levels, outer-joined, target cleaned
    eia: pd.DataFrame                         # WTI futures contracts 1-4 (may be empty / stale)
    weekly: Dict[str, Tuple[pd.Series, int]]  # name -> (series by period date, release lag in days)


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, index_col=0, parse_dates=True).sort_index().apply(pd.to_numeric, errors="coerce")


def load_inputs(cfg: Config, raw_dir: Path = RAW_DIR) -> Inputs:
    yahoo = _read(raw_dir / "yahoo_daily.csv")
    extra = _read(raw_dir / "yahoo_extra_daily.csv")
    fred = _read(raw_dir / "fred_daily.csv")
    if yahoo.empty:
        raise FileNotFoundError(f"{raw_dir / 'yahoo_daily.csv'} not found: run scripts/update_data.py first")

    fd = {n: fred[n].dropna().sort_index().shift(cfg.fred_lag) for n in FRED_DAILY if n in fred.columns}
    extra = extra.drop(columns=[c for c in extra.columns if c in yahoo.columns])   # tolerate one merged file
    prices = yahoo.join(extra, how="outer").join(pd.DataFrame(fd), how="outer").sort_index()
    prices = prices.drop(columns=[c for c in cfg.exclude if c in prices.columns])
    if cfg.target not in prices.columns:
        raise KeyError(f"target {cfg.target!r} not in {list(prices.columns)}")
    p = prices[cfg.target]
    prices[cfg.target] = p.where(p > 0)                      # non-positive prices -> NaN

    wk = _read(raw_dir / "weekly_raw.csv")
    weekly = {n: (wk[n].dropna(), cfg.weekly_release_lag) for n in WEEKLY if n in wk.columns}
    return Inputs(prices=prices, eia=_read(raw_dir / "eia_wti_futures.csv"), weekly=weekly)


def truncate_inputs(inputs: Inputs, date) -> Inputs:
    """Inputs as they were known on `date` (everything after it removed). Weekly series keep their release-lag
    logic, so a value dated before `date` but released after it is still not used by the features."""
    d = pd.Timestamp(date)
    return Inputs(prices=inputs.prices.loc[:d],
                  eia=inputs.eia.loc[:d] if len(inputs.eia) else inputs.eia,
                  weekly={k: (s.loc[:d], lag) for k, (s, lag) in inputs.weekly.items()})


def target_inputs(inputs: Inputs, cfg: Config, target: str) -> Inputs:
    """Inputs prepared for forecasting `target`: its own price column is cleaned (non-positive prices -> missing) and cut
    to cfg.series_start[target]; every other column is left as it is."""
    px = inputs.prices.copy()
    s = px[target].where(px[target] > 0)
    start = cfg.series_start.get(target)
    if start:
        s = s.where(s.index >= pd.Timestamp(start))
    px[target] = s
    return Inputs(prices=px, eia=inputs.eia, weekly=inputs.weekly)
