"""Download the last 90 days of every series and merge them into the CSVs in Data/raw/.

* Existing history is kept; for dates in the overlap window the newly downloaded values win
  (so revised data is corrected) unless the new value is missing.
* A file that does not exist yet (or lacks a series) is downloaded in full from START_DATE.
* Writes update_status.json (read by send_email.py) listing any series that failed to download.
* Exit code 1 only if nothing at all could be downloaded.

Usage:  python scripts/update_data.py            (incremental, last 90 days)
        FULL_REFRESH=1 python scripts/update_data.py   (re-download everything)
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
FRED_SERIES = {"wti_spot": "DCOILWTICO", "brent_spot": "DCOILBRENTEU",
               "ust10y": "DGS10", "ust2y": "DGS2", "curve_10y2y": "T10Y2Y",
               "breakeven10y": "T10YIE", "ovx": "OVXCLS"}


def download_yahoo(start, failed):
    end = (pd.Timestamp.today() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    cols = {}
    for name, ticker in YF_TICKERS.items():
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


def download_fred(start, failed):
    cols = {}
    for name, series_id in FRED_SERIES.items():
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


def update_csv(downloader, path, expected_cols, failed):
    old, start = None, START_DATE
    if path.exists() and not FULL_REFRESH:
        old = pd.read_csv(path, index_col=0, parse_dates=True)
        if set(expected_cols) <= set(old.columns):
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
    failed = []
    try:
        yahoo = update_csv(download_yahoo, RAW / "yahoo_daily.csv", YF_TICKERS, failed)
        fred = update_csv(download_fred, RAW / "fred_daily.csv", FRED_SERIES, failed)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    status = {"updated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "failed_series": failed,
              "yahoo_last_date": str(yahoo.index.max().date()), "fred_last_date": str(fred.index.max().date())}
    (ROOT / "update_status.json").write_text(json.dumps(status, indent=2))
    print("Done:", status)
    if len(failed) == len(YF_TICKERS) + len(FRED_SERIES):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
