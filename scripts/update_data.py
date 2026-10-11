"""Download the latest data for every series and merge it into the CSVs in Data/raw/.

Files written (all keep full history; existing values are kept, new values win):
    yahoo_daily.csv        Yahoo futures / indices (unchanged column set)
    yahoo_extra_daily.csv  Yahoo extras used by the forecasting models (USDCAD, USDNOK, EEM)
    fred_daily.csv         FRED daily series (spot prices, rates, OVX, 10Y-3M, EPU)
    weekly_raw.csv         Weekly series: NY Fed WEI (FRED) and US crude stocks (EIA)
    eia_wti_futures.csv    EIA WTI futures contracts 1-4 (EIA's files stop in April 2024)

Daily files are refreshed incrementally (last OVERLAP_DAYS); a file or column that is missing is
downloaded in full from START_DATE. The small weekly/EIA files are always re-downloaded in full.

Writes update_status.json (read by the e-mail step) with any series that failed.
Exit code 1 only if nothing at all could be downloaded.

Usage:  python scripts/update_data.py
        FULL_REFRESH=1 python scripts/update_data.py     (re-download everything)
"""
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "Data" / "raw"
START_DATE = "1980-01-01"          # used for first-time / full downloads
OVERLAP_DAYS = 90
FULL_REFRESH = bool(os.getenv("FULL_REFRESH", ""))

YF_TICKERS = {"wti_fut": "CL=F", "brent_fut": "BZ=F", "gasoline": "RB=F", "heating_oil": "HO=F",
              "natgas": "NG=F", "xle": "XLE", "sp500": "^GSPC", "vix": "^VIX",
              "dxy": "DX-Y.NYB", "gold": "GC=F", "copper": "HG=F"}
YF_EXTRA = {"usdcad": "CAD=X", "usdnok": "NOK=X", "eem": "EEM"}
FRED_SERIES = {"wti_spot": "DCOILWTICO", "brent_spot": "DCOILBRENTEU",
               "ust10y": "DGS10", "ust2y": "DGS2", "curve_10y2y": "T10Y2Y",
               "breakeven10y": "T10YIE", "ovx": "OVXCLS",
               "t10y3m": "T10Y3M", "epu": "USEPUINDXD"}
FRED_WEEKLY = {"wei": "WEI"}                                   # FRED weekly (period date = week end)
EIA_WEEKLY = {"crude_stocks": "WCESTUS1w"}                     # EIA weekly US crude stocks excl. SPR
EIA_FUTURES = {"wti_c1": "RCLC1d", "wti_c2": "RCLC2d", "wti_c3": "RCLC3d", "wti_c4": "RCLC4d"}
EIA_URL = "https://www.eia.gov/dnav/pet/hist_xls/{sid}.xls"


# --------------------------------------------------------------------------- downloaders
def _yahoo(tickers, start, failed):
    end = (pd.Timestamp.today() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    cols = {}
    for name, ticker in tickers.items():
        try:
            x = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False, threads=False)
            if isinstance(x.columns, pd.MultiIndex):
                x.columns = x.columns.get_level_values(0)
            if x.empty:
                raise ValueError("empty response")
            s = x["Close"].copy()
            s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
            cols[name] = s
            print(f"  Yahoo {name:<12} {ticker:<9} {len(s):>5} rows")
        except Exception as exc:
            failed.append(name)
            print(f"  WARNING: Yahoo {name} failed: {exc}")
    return pd.DataFrame(cols).sort_index()


def _fred(series, start, failed):
    cols = {}
    for name, series_id in series.items():
        try:
            x = pd.read_csv(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start}")
            s = pd.to_numeric(x.iloc[:, 1], errors="coerce")
            s.index = pd.to_datetime(x.iloc[:, 0])
            cols[name] = s
            print(f"  FRED  {name:<12} {series_id:<13} {s.notna().sum():>5} rows")
        except Exception as exc:
            failed.append(name)
            print(f"  WARNING: FRED {name} failed: {exc}")
    return pd.DataFrame(cols).sort_index()


def _eia(series, failed):
    cols = {}
    for name, sid in series.items():
        try:
            d = pd.read_excel(EIA_URL.format(sid=sid), sheet_name="Data 1", skiprows=2)   # needs xlrd
            s = pd.Series(pd.to_numeric(d.iloc[:, 1], errors="coerce").to_numpy(), index=pd.to_datetime(d.iloc[:, 0]))
            cols[name] = s.dropna().sort_index()
            print(f"  EIA   {name:<12} {sid:<10} {len(cols[name]):>5} rows, last {cols[name].index.max().date()}")
        except Exception as exc:
            failed.append(name)
            print(f"  WARNING: EIA {name} failed: {exc}")
    return pd.DataFrame(cols).sort_index()


# --------------------------------------------------------------------------- merge
def update_csv(path, downloader, expected_cols, failed, incremental=True):
    """downloader(start, failed) -> DataFrame. Merge into path; keep old values where new ones are missing."""
    old, start = None, START_DATE
    if path.exists() and not FULL_REFRESH:
        old = pd.read_csv(path, index_col=0, parse_dates=True)
        if incremental and set(expected_cols) <= set(old.columns):
            start = (old.index.max() - pd.Timedelta(days=OVERLAP_DAYS)).strftime("%Y-%m-%d")
    print(f"{path.name}: downloading from {start}")
    new = downloader(start, failed)
    if new.empty and old is None:
        raise RuntimeError(f"Nothing downloaded and no existing file for {path.name}")
    out = new.combine_first(old) if old is not None and not new.empty else (old if old is not None else new)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out.index.name = "date"
    out.to_csv(path)
    return out


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    failed, results = [], {}
    jobs = [
        ("yahoo_daily.csv", lambda s, f: _yahoo(YF_TICKERS, s, f), YF_TICKERS, True),
        ("yahoo_extra_daily.csv", lambda s, f: _yahoo(YF_EXTRA, s, f), YF_EXTRA, True),
        ("fred_daily.csv", lambda s, f: _fred(FRED_SERIES, s, f), FRED_SERIES, True),
        ("weekly_raw.csv", lambda s, f: pd.concat([_fred(FRED_WEEKLY, START_DATE, f), _eia(EIA_WEEKLY, f)], axis=1).sort_index(),
         {**FRED_WEEKLY, **EIA_WEEKLY}, False),
        ("eia_wti_futures.csv", lambda s, f: _eia(EIA_FUTURES, f), EIA_FUTURES, False),
    ]
    for fname, downloader, expected, incremental in jobs:
        try:
            results[fname] = update_csv(RAW / fname, downloader, expected, failed, incremental)
        except Exception as exc:
            print(f"ERROR: {fname}: {exc}", file=sys.stderr)
            failed.extend(list(expected))
    if not results:
        return 1
    status = {"updated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "failed_series": sorted(set(failed)),
              "last_dates": {f: str(df.index.max().date()) for f, df in results.items()}}
    (ROOT / "update_status.json").write_text(json.dumps(status, indent=2))
    print("Done:", json.dumps(status))
    return 0


if __name__ == "__main__":
    sys.exit(main())
