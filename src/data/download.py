"""
Automated data download from yfinance and FRED API.

CBOE data (VIX9D, VIX3M, VIX6M, put/call ratio) and Kaggle options data
must be downloaded manually — see README.md for instructions.
"""

import os
import pandas as pd
import yfinance as yf

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")


def download_yfinance(start="2010-01-01", end="2024-01-01", data_dir=None):
    """
    Download SPY OHLCV, VIX, VVIX, HYG, LQD, and DXY from yfinance.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keyed by ticker symbol.
    """
    data_dir = data_dir or os.path.abspath(DATA_DIR)
    os.makedirs(data_dir, exist_ok=True)

    tickers = {
        "SPY": "SPY",
        "VIX": "^VIX",
        "VVIX": "^VVIX",
        "HYG": "HYG",
        "LQD": "LQD",
        "DXY": "DX-Y.NYB",
    }

    results = {}
    for name, ticker in tickers.items():
        print(f"Downloading {name} ({ticker})...")
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        path = os.path.join(data_dir, f"{name.lower()}.csv")
        df.to_csv(path)
        results[name] = df
        print(f"  -> {len(df)} rows saved to {path}")

    return results


def download_fred(start="2010-01-01", end="2024-01-01", api_key=None, data_dir=None):
    """
    Download DGS2 and DGS10 treasury yields from FRED.

    Parameters
    ----------
    api_key : str or None
        FRED API key. If None, reads from FRED_API_KEY env var.

    Returns
    -------
    pd.DataFrame
        Columns: treasury_2y, treasury_10y. Indexed by date.
    """
    data_dir = data_dir or os.path.abspath(DATA_DIR)
    os.makedirs(data_dir, exist_ok=True)

    if api_key is None:
        api_key = os.environ.get("FRED_API_KEY")

    if api_key is None:
        print("WARNING: No FRED API key found. Attempting CSV fallback...")
        return _fred_fallback(data_dir)

    from fredapi import Fred
    fred = Fred(api_key=api_key)

    series_map = {"DGS2": "treasury_2y", "DGS10": "treasury_10y"}
    frames = {}
    for series_id, col_name in series_map.items():
        print(f"Downloading {series_id} from FRED...")
        s = fred.get_series(series_id, observation_start=start, observation_end=end)
        frames[col_name] = s

    df = pd.DataFrame(frames)
    df.index.name = "Date"
    # FRED reports percentages; convert to decimal
    df = df / 100.0

    path = os.path.join(data_dir, "treasury.csv")
    df.to_csv(path)
    print(f"  -> {len(df)} rows saved to {path}")
    return df


def _fred_fallback(data_dir):
    """Try loading treasury data from existing CSV if FRED API key unavailable."""
    path = os.path.join(data_dir, "treasury.csv")
    if os.path.exists(path):
        print(f"  Loading from existing {path}")
        return pd.read_csv(path, index_col=0, parse_dates=True)
    raise FileNotFoundError(
        "No FRED API key and no treasury.csv found. "
        "Set FRED_API_KEY env var or place treasury.csv in data/raw/."
    )


def load_cboe_files(data_dir=None):
    """
    Load manually-downloaded CBOE CSV files.

    Expected files in data_dir:
        vix9d.csv   — VIX9D historical data
        vix3m.csv   — VIX3M historical data
        vix6m.csv   — VIX6M historical data
        put_call_ratio.csv — CBOE total put/call ratio

    Returns
    -------
    dict[str, pd.Series]
        Each series indexed by date.
    """
    data_dir = data_dir or os.path.abspath(DATA_DIR)

    result = {}
    for name, filename in [("vix9d", "vix9d.csv"), ("vix3m", "vix3m.csv"),
                           ("vix6m", "vix6m.csv"),
                           ("put_call_ratio", "put_call_ratio.csv")]:
        path = os.path.join(data_dir, filename)
        if not os.path.exists(path):
            print(f"WARNING: {path} not found. This must be downloaded manually from CBOE.")
            result[name] = None
            continue

        df = pd.read_csv(path)
        # CBOE files vary in format; try common column names
        date_col = None
        for candidate in ["DATE", "Date", "date", "Trade Date"]:
            if candidate in df.columns:
                date_col = candidate
                break

        val_col = None
        for candidate in ["CLOSE", "Close", "close", name.upper(),
                          "P/C Ratio", "Total Put/Call Ratio", "TOTAL P/C RATIO"]:
            if candidate in df.columns:
                val_col = candidate
                break

        if date_col is None or val_col is None:
            print(f"WARNING: Could not parse {filename}. Columns: {list(df.columns)}")
            result[name] = None
            continue

        s = pd.Series(
            pd.to_numeric(df[val_col], errors="coerce").values,
            index=pd.to_datetime(df[date_col]),
            name=name,
        )
        s = s.dropna()
        result[name] = s
        print(f"  Loaded {name}: {len(s)} rows")

    return result


def load_options_data(data_dir=None):
    """
    Load the Kaggle SPY options dataset.

    Expected file: spy_options.csv (or similar) in data_dir.

    Returns
    -------
    pd.DataFrame
    """
    data_dir = data_dir or os.path.abspath(DATA_DIR)

    # Try common filenames
    for fname in ["spy_options.csv", "spy_options_data.csv",
                  "sp500_options.csv", "options.csv"]:
        path = os.path.join(data_dir, fname)
        if os.path.exists(path):
            print(f"Loading options data from {path}...")
            df = pd.read_csv(path, parse_dates=["date", "expiration"])
            print(f"  -> {len(df)} rows")
            return df

    raise FileNotFoundError(
        "Options CSV not found in data/raw/. Download from Kaggle and place there."
    )


if __name__ == "__main__":
    print("=" * 60)
    print("Downloading market data...")
    print("=" * 60)
    download_yfinance()
    print()
    download_fred()
    print()
    print("Remember to manually download CBOE and Kaggle files.")
