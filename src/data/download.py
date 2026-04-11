"""
Automated data download from yfinance, FRED API, and CBOE.

Only the Kaggle SPY options dataset must be downloaded manually.
"""

import os
import urllib.request
import pandas as pd
import yfinance as yf

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")

# CBOE direct-download CSV URLs (public, no auth needed)
CBOE_TOTALPC_URL = "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/totalpc.csv"


def download_yfinance(start="2010-01-01", end="2024-01-01", data_dir=None):
    """
    Download SPY OHLCV, VIX, VVIX, HYG, LQD, DXY, VIX9D, VIX3M, VIX6M from yfinance.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keyed by name.
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
        "VIX9D": "^VIX9D",
        "VIX3M": "^VIX3M",
        "VIX6M": "^VIX6M",
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


def download_cboe_put_call_ratio(data_dir=None):
    """
    Download CBOE Total Put/Call Ratio CSV (covers 2006-2019).

    For 2020-2023 test period, values are forward-filled during preprocessing.

    Returns
    -------
    pd.DataFrame
        Put/call ratio data.
    """
    data_dir = data_dir or os.path.abspath(DATA_DIR)
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "put_call_ratio.csv")

    print("Downloading CBOE Total Put/Call Ratio...")
    urllib.request.urlretrieve(CBOE_TOTALPC_URL, path)

    # CBOE CSV has disclaimer rows before actual data — find the header row
    with open(path, "r") as f:
        lines = f.readlines()

    header_idx = None
    for i, line in enumerate(lines):
        if "DATE" in line.upper() and ("P/C" in line.upper() or "RATIO" in line.upper()):
            header_idx = i
            break
        if "TRADE DATE" in line.upper():
            header_idx = i
            break

    if header_idx is not None:
        df = pd.read_csv(path, skiprows=header_idx)
    else:
        # Try skipping rows until we find numeric data
        for skip in range(10):
            try:
                df = pd.read_csv(path, skiprows=skip)
                # Check if first column looks like dates
                first_col = df.columns[0]
                pd.to_datetime(df[first_col].iloc[0])
                break
            except (ValueError, TypeError):
                continue
        else:
            df = pd.read_csv(path)

    # Clean up column names
    df.columns = [c.strip() for c in df.columns]
    print(f"  -> {len(df)} rows saved to {path}")
    print(f"  Columns: {list(df.columns)}")
    return df


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
        print("WARNING: No FRED API key found. Attempting yfinance fallback...")
        return _fred_fallback(data_dir, start=start, end=end)

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


def _fred_fallback(data_dir, start="2010-01-01", end="2024-01-01"):
    """Try loading treasury data from existing CSV, or download from yfinance."""
    path = os.path.join(data_dir, "treasury.csv")
    if os.path.exists(path):
        print(f"  Loading from existing {path}")
        return pd.read_csv(path, index_col=0, parse_dates=True)

    # yfinance fallback: ^TNX (10y yield), ^IRX (13-week bill yield as 2y proxy)
    print("  Downloading treasury yields from yfinance (^TNX, ^IRX)...")
    tnx = yf.download("^TNX", start=start, end=end, auto_adjust=True, progress=False)
    irx = yf.download("^IRX", start=start, end=end, auto_adjust=True, progress=False)

    if isinstance(tnx.columns, pd.MultiIndex):
        tnx.columns = tnx.columns.droplevel(1)
    if isinstance(irx.columns, pd.MultiIndex):
        irx.columns = irx.columns.droplevel(1)

    # These indices report yields in percentage points; convert to decimal
    df = pd.DataFrame({
        "treasury_2y": irx["Close"] / 100.0,
        "treasury_10y": tnx["Close"] / 100.0,
    })
    df.index.name = "Date"
    df = df.dropna()

    df.to_csv(path)
    print(f"  -> {len(df)} rows saved to {path}")
    return df


def load_cboe_files(data_dir=None):
    """
    Load CBOE data: VIX9D/VIX3M/VIX6M from yfinance CSVs, put/call ratio from CBOE CSV.

    Returns
    -------
    dict[str, pd.Series]
        Each series indexed by date.
    """
    data_dir = data_dir or os.path.abspath(DATA_DIR)
    result = {}

    # VIX term structure — from yfinance CSVs (downloaded in download_yfinance)
    for name in ["vix9d", "vix3m", "vix6m"]:
        path = os.path.join(data_dir, f"{name}.csv")
        if not os.path.exists(path):
            print(f"WARNING: {path} not found. Run download_yfinance() first.")
            result[name] = None
            continue
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        # Extract Close column
        close_col = None
        for c in ["Close", "close", "Adj Close"]:
            if c in df.columns:
                close_col = c
                break
        if close_col is None and len(df.columns) > 0:
            close_col = df.columns[0]
        s = df[close_col].dropna()
        s.name = name
        result[name] = s
        print(f"  Loaded {name}: {len(s)} rows")

    # Put/call ratio — from CBOE CSV
    pcr_path = os.path.join(data_dir, "put_call_ratio.csv")
    if not os.path.exists(pcr_path):
        print(f"WARNING: {pcr_path} not found. Run download_cboe_put_call_ratio() first.")
        result["put_call_ratio"] = None
    else:
        df = pd.read_csv(pcr_path)
        # CBOE totalpc.csv columns: DATE, CALLS, PUTS, TOTAL, P/C Ratio
        date_col = None
        for candidate in ["DATE", "Date", "date", "Trade Date"]:
            if candidate in df.columns:
                date_col = candidate
                break
        val_col = None
        for candidate in ["P/C Ratio", "P/C RATIO", "Total Put/Call Ratio",
                          "TOTAL P/C RATIO", "put_call_ratio"]:
            if candidate in df.columns:
                val_col = candidate
                break

        if date_col and val_col:
            s = pd.Series(
                pd.to_numeric(df[val_col], errors="coerce").values,
                index=pd.to_datetime(df[date_col]),
                name="put_call_ratio",
            ).dropna()
            result["put_call_ratio"] = s
            print(f"  Loaded put_call_ratio: {len(s)} rows ({s.index[0].date()} to {s.index[-1].date()})")
        else:
            print(f"WARNING: Could not parse put_call_ratio.csv. Columns: {list(df.columns)}")
            result["put_call_ratio"] = None

    return result


def _normalize_options_columns(df):
    """Strip brackets, lowercase, and normalize column names from various Kaggle formats."""
    # Strip brackets and whitespace, lowercase
    df.columns = [c.strip().lower().strip("[]") for c in df.columns]

    # Parse date columns
    for dcol in ["date", "quote_date", "trade_date", "data_date"]:
        if dcol in df.columns:
            df["date"] = pd.to_datetime(df[dcol])
            break
    for dcol in ["expiration", "expire_date", "expiry", "expiration_date"]:
        if dcol in df.columns:
            df["expiration"] = pd.to_datetime(df[dcol])
            break

    return df


def load_options_data(data_dir=None):
    """
    Load the Kaggle SPY options dataset.

    Handles multiple possible file formats and column naming conventions.

    Returns
    -------
    pd.DataFrame
    """
    data_dir = data_dir or os.path.abspath(DATA_DIR)

    # Try common filenames
    for fname in ["spy_options.csv", "spy_options_data.csv",
                  "sp500_options.csv", "options.csv",
                  "spy_options_eod.csv", "cleaned_options_data.csv"]:
        path = os.path.join(data_dir, fname)
        if os.path.exists(path):
            print(f"Loading options data from {path}...")
            # Try to detect date columns
            df = pd.read_csv(path, low_memory=False)
            df = _normalize_options_columns(df)
            print(f"  -> {len(df)} rows, columns: {list(df.columns)[:10]}...")
            return df

    # Also check for any CSV with "option" in the name
    for fname in sorted(os.listdir(data_dir)):
        if fname.endswith(".csv") and "option" in fname.lower():
            path = os.path.join(data_dir, fname)
            print(f"Found options CSV by name: {path}")
            df = pd.read_csv(path, low_memory=False)
            df = _normalize_options_columns(df)
            print(f"  -> {len(df)} rows, columns: {list(df.columns)[:10]}...")
            return df

    # Also check for subdirectories (Kaggle zips sometimes extract into folders)
    for subdir in os.listdir(data_dir):
        subpath = os.path.join(data_dir, subdir)
        if os.path.isdir(subpath):
            for fname in os.listdir(subpath):
                if fname.endswith(".csv"):
                    path = os.path.join(subpath, fname)
                    print(f"Found CSV in subdirectory: {path}")
                    df = pd.read_csv(path, low_memory=False)
                    df = _normalize_options_columns(df)
                    print(f"  -> {len(df)} rows")
                    return df

    raise FileNotFoundError(
        "Options CSV not found in data/raw/. "
        "Download from https://www.kaggle.com/datasets/benjaminbtang/spy-options-2010-2023-eod "
        "and place the CSV in data/raw/"
    )


def download_all(start="2010-01-01", end="2024-01-01", fred_api_key=None, data_dir=None):
    """Download all automated sources in one call."""
    data_dir = data_dir or os.path.abspath(DATA_DIR)
    print("=" * 60)
    print("Step 1/3: yfinance (SPY, VIX, VVIX, HYG, LQD, DXY, VIX9D, VIX3M, VIX6M)")
    print("=" * 60)
    download_yfinance(start=start, end=end, data_dir=data_dir)

    print("\n" + "=" * 60)
    print("Step 2/3: CBOE Put/Call Ratio")
    print("=" * 60)
    download_cboe_put_call_ratio(data_dir=data_dir)

    print("\n" + "=" * 60)
    print("Step 3/3: FRED Treasury Yields")
    print("=" * 60)
    download_fred(start=start, end=end, api_key=fred_api_key, data_dir=data_dir)

    print("\n" + "=" * 60)
    print("DONE. Only manual download needed: Kaggle SPY options data.")
    print("  https://www.kaggle.com/datasets/benjaminbtang/spy-options-2010-2023-eod")
    print("  Place the CSV in data/raw/")
    print("=" * 60)


if __name__ == "__main__":
    download_all()
