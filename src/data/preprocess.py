"""
Feature engineering, normalization, sliding-window construction, and options preprocessing.

All rolling features use .shift(1) before the rolling window to prevent lookahead bias.
Scaler is fit on training data (2010-2019) only.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# The 20 features used throughout the pipeline (GAN trains on all 20)
FEATURE_COLS = [
    "log_return", "rv_5d", "rv_21d", "rv_63d",
    "gk_vol", "vix", "vix_slope", "vix_6m_slope",
    "vvix", "rv_iv_spread", "momentum_5d", "momentum_21d",
    "skew_21d", "kurt_21d", "credit_spread", "volume_zscore",
    "treasury_2y", "put_call_ratio", "dxy", "vol_regime",
]

SEQ_LEN = 60  # trading days per sequence (~3 months) — used by GAN

# Run 3 pricing-network config: drop the four macro features that ablation
# A2 flagged as harmful noise (credit_spread idx 14, treasury_2y 16,
# put_call_ratio 17, dxy 18), and shorten the context window to 30 days
# (ablation A3 showed 60 was too long). The GAN still trains on all 20
# features at seq_len=60 — these drops apply only at the Transformer stage.
MACRO_DROP_INDICES = [14, 16, 17, 18]
PRICING_SEQ_LEN = 30


def prepare_pricing_sequences(sequences, drop_macros=True, seq_len=PRICING_SEQ_LEN):
    """
    Slice full (N, 60, 20) sequences down to the Transformer's input shape.

    Parameters
    ----------
    sequences : np.ndarray, shape (N, src_seq_len, 20)
        Real or GAN-synthesized market sequences.
    drop_macros : bool
        Drop the four macro features identified as noise by ablation A2.
    seq_len : int
        Context window length. Takes the *last* seq_len steps so the most
        recent history (which is what the pricing model actually uses) is
        preserved.

    Returns
    -------
    np.ndarray, shape (N, seq_len, 20 - len(dropped))
    """
    out = sequences
    if seq_len is not None and seq_len < out.shape[1]:
        out = out[:, -seq_len:, :]
    if drop_macros:
        out = np.delete(out, MACRO_DROP_INDICES, axis=2)
    return out


# After MACRO_DROP_INDICES has been removed, the remaining 16 features are
# reindexed. Positions in the reduced set we care about for A1/A6:
#   original 6  vix_slope        → reduced 6   (unchanged, macros all come later)
#   original 7  vix_6m_slope     → reduced 7
#   original 15 volume_zscore    → reduced 14
#   original 19 vol_regime       → reduced 15
REDUCED_VIX_SLOPE_INDICES = [6, 7]
REDUCED_VOL_REGIME_INDEX = 15


def prepare_ablation_sequences(sequences, ablation_id):
    """
    Build the input sequences for a specific ablation study, starting from
    the raw (N, 60, 20) arrays.

    Each ablation is expressed as a delta from the Run 3 baseline
    (drop_macros=True, seq_len=30). Model input_dim and seq_len for each
    variant are produced by build_ablation_transformer in src/models/transformer.py.

    Parameters
    ----------
    sequences : np.ndarray, shape (N, 60, 20)
        Raw sequences with all 20 features and full 60-step context.
    ablation_id : str
        One of A1..A6.

    Returns
    -------
    np.ndarray
        Prepared sequences for that ablation.
    """
    if ablation_id == "A1":
        # Drop VIX slope features on top of baseline
        out = prepare_pricing_sequences(sequences)  # (N, 30, 16)
        return np.delete(out, REDUCED_VIX_SLOPE_INDICES, axis=2)  # (N, 30, 14)
    if ablation_id == "A2":
        # Keep macros (compare to baseline which drops them)
        return prepare_pricing_sequences(sequences, drop_macros=False)  # (N, 30, 20)
    if ablation_id == "A3":
        return prepare_pricing_sequences(sequences, seq_len=20)  # (N, 20, 16)
    if ablation_id == "A4":
        return prepare_pricing_sequences(sequences, seq_len=60)  # (N, 60, 16)
    if ablation_id == "A5":
        # A5 is a model-depth ablation; features/seq_len match baseline
        return prepare_pricing_sequences(sequences)
    if ablation_id == "A6":
        out = prepare_pricing_sequences(sequences)  # (N, 30, 16)
        return np.delete(out, [REDUCED_VOL_REGIME_INDEX], axis=2)  # (N, 30, 15)
    raise ValueError(f"Unknown ablation ID: {ablation_id}")

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")


# ---------------------------------------------------------------------------
# Step 1-2: Build master DataFrame and engineer features
# ---------------------------------------------------------------------------

def build_master_dataframe(spy, vix, vvix, hyg, lqd, dxy, treasury, cboe):
    """
    Merge all raw data sources into a single daily DataFrame aligned to SPY trading days,
    then engineer the 20 features.

    Parameters
    ----------
    spy : pd.DataFrame
        SPY OHLCV with DatetimeIndex.
    vix, vvix, hyg, lqd, dxy : pd.Series or pd.DataFrame
        Close prices, DatetimeIndex.
    treasury : pd.DataFrame
        Columns treasury_2y, treasury_10y.
    cboe : dict
        Keys: vix9d, vix3m, vix6m, put_call_ratio (pd.Series each).

    Returns
    -------
    pd.DataFrame
        Master DataFrame with 20 feature columns, indexed by SPY trading days.
    """
    # Normalize column casing
    spy_cols = {c.lower(): c for c in spy.columns}
    close = spy[spy_cols.get("close", "Close")].copy()
    opn = spy[spy_cols.get("open", "Open")].copy()
    high = spy[spy_cols.get("high", "High")].copy()
    low = spy[spy_cols.get("low", "Low")].copy()
    volume = spy[spy_cols.get("volume", "Volume")].copy()

    idx = close.index  # SPY trading days

    # Helper: reindex to SPY days and forward-fill
    def align(series, name):
        if series is None:
            return pd.Series(np.nan, index=idx, name=name)
        if isinstance(series, pd.DataFrame):
            if name in series.columns:
                series = series[name]
            else:
                series = series.iloc[:, 0]
        s = series.reindex(idx).ffill()
        s.name = name
        return s

    # Extract close prices from DataFrames if needed
    def _close(obj):
        if obj is None:
            return None
        if isinstance(obj, pd.DataFrame):
            for c in ["Close", "close", "Adj Close"]:
                if c in obj.columns:
                    return obj[c]
            return obj.iloc[:, 0]
        return obj

    vix_s = align(_close(vix), "vix")
    vvix_s = align(_close(vvix), "vvix")
    hyg_s = align(_close(hyg), "hyg_close")
    lqd_s = align(_close(lqd), "lqd_close")
    dxy_s = align(_close(dxy), "dxy")

    # Treasury yields
    t2y = align(treasury, "treasury_2y") if treasury is not None else pd.Series(np.nan, index=idx, name="treasury_2y")
    if isinstance(t2y, pd.DataFrame):
        t2y = t2y["treasury_2y"] if "treasury_2y" in t2y.columns else t2y.iloc[:, 0]
    t2y = t2y.reindex(idx).ffill()

    # CBOE data
    vix9d_s = align(cboe.get("vix9d"), "vix9d") if cboe else pd.Series(np.nan, index=idx, name="vix9d")
    vix3m_s = align(cboe.get("vix3m"), "vix3m") if cboe else pd.Series(np.nan, index=idx, name="vix3m")
    vix6m_s = align(cboe.get("vix6m"), "vix6m") if cboe else pd.Series(np.nan, index=idx, name="vix6m")
    pcr_s = align(cboe.get("put_call_ratio"), "put_call_ratio") if cboe else pd.Series(np.nan, index=idx, name="put_call_ratio")

    # ── Feature engineering (Rule 5: .shift(1) before all rolling) ──

    # Log return: uses close[t-1] and close[t-2] due to shift
    log_return = np.log(close / close.shift(1))

    # Realized volatility (annualized) — shift log_return by 1 before rolling
    lr_shifted = log_return.shift(1)
    rv_5d = lr_shifted.rolling(5).std() * np.sqrt(252)
    rv_21d = lr_shifted.rolling(21).std() * np.sqrt(252)
    rv_63d = lr_shifted.rolling(63).std() * np.sqrt(252)

    # Garman-Klass volatility — use previous day's OHLC
    gk_inner = (
        0.5 * np.log(high.shift(1) / low.shift(1)) ** 2
        - (2 * np.log(2) - 1) * np.log(close.shift(1) / opn.shift(1)) ** 2
    )
    gk_vol = np.sqrt(gk_inner.rolling(21).mean()) * np.sqrt(252)

    # VIX term structure
    vix_slope = vix3m_s - vix9d_s
    vix_6m_slope = vix6m_s - vix3m_s

    # Implied vs realized spread
    rv_iv_spread = vix_s - rv_21d

    # Momentum
    momentum_5d = lr_shifted.rolling(5).sum()
    momentum_21d = lr_shifted.rolling(21).sum()

    # Higher moments
    skew_21d = lr_shifted.rolling(21).skew()
    kurt_21d = lr_shifted.rolling(21).kurt()

    # Market stress
    credit_spread = hyg_s / lqd_s
    vol_shifted = volume.shift(1)
    volume_zscore = (
        (vol_shifted - vol_shifted.rolling(21).mean()) / vol_shifted.rolling(21).std()
    )

    # Regime label
    vol_regime = pd.Series(1, index=idx, name="vol_regime", dtype=float)
    vol_regime[vix_s < 15] = 0
    vol_regime[vix_s > 25] = 2

    # Assemble
    master = pd.DataFrame({
        "log_return": log_return,
        "rv_5d": rv_5d,
        "rv_21d": rv_21d,
        "rv_63d": rv_63d,
        "gk_vol": gk_vol,
        "vix": vix_s,
        "vix_slope": vix_slope,
        "vix_6m_slope": vix_6m_slope,
        "vvix": vvix_s,
        "rv_iv_spread": rv_iv_spread,
        "momentum_5d": momentum_5d,
        "momentum_21d": momentum_21d,
        "skew_21d": skew_21d,
        "kurt_21d": kurt_21d,
        "credit_spread": credit_spread,
        "volume_zscore": volume_zscore,
        "treasury_2y": t2y,
        "put_call_ratio": pcr_s,
        "dxy": dxy_s,
        "vol_regime": vol_regime,
    }, index=idx)

    # Also keep raw close for options preprocessing
    master["spy_close"] = close

    return master


# ---------------------------------------------------------------------------
# Steps 3-6: Drop NaN, split, normalize, build sequences
# ---------------------------------------------------------------------------

def clean_and_split(master, train_end="2019-12-31", test_start="2020-01-01"):
    """Forward-fill remaining NaN, drop any rows still incomplete, and split."""
    # Forward-fill features that may have gaps (e.g. put_call_ratio ends 2019)
    master[FEATURE_COLS] = master[FEATURE_COLS].ffill()
    # Back-fill early rows that have NaN from rolling window warm-up
    master[FEATURE_COLS] = master[FEATURE_COLS].bfill()
    master = master.dropna(subset=FEATURE_COLS)
    train_df = master.loc[:train_end].copy()
    test_df = master.loc[test_start:].copy()
    return train_df, test_df


def fit_scaler(train_df, save_path=None):
    """Fit StandardScaler on training features only. Optionally save to disk."""
    scaler = StandardScaler()
    scaler.fit(train_df[FEATURE_COLS].values)

    if save_path is None:
        save_path = os.path.join(os.path.abspath(PROCESSED_DIR), "scaler.pkl")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {save_path}")
    return scaler


def scale_data(df, scaler):
    """Apply pre-fit scaler to feature columns. Returns numpy array."""
    return scaler.transform(df[FEATURE_COLS].values)


def build_sequences(scaled_array, seq_len=SEQ_LEN):
    """
    Build sliding-window sequences for GAN training.

    Parameters
    ----------
    scaled_array : np.ndarray, shape (T, 20)
    seq_len : int

    Returns
    -------
    np.ndarray, shape (T - seq_len, seq_len, 20)
    """
    sequences = []
    for i in range(len(scaled_array) - seq_len):
        sequences.append(scaled_array[i: i + seq_len])
    return np.array(sequences, dtype=np.float32)


# ---------------------------------------------------------------------------
# Step 7: Options data preprocessing
# ---------------------------------------------------------------------------

def preprocess_options(opts_df, master_df, scaler, seq_len=SEQ_LEN):
    """
    Filter and enrich options data with market sequences and t-1 contract features.

    Parameters
    ----------
    opts_df : pd.DataFrame
        Raw Kaggle options data.
    master_df : pd.DataFrame
        Full master DataFrame (unscaled, with spy_close column).
    scaler : StandardScaler
        Pre-fit scaler.
    seq_len : int
        Sequence length.

    Returns
    -------
    (pd.DataFrame, np.ndarray)
        opts_df: Filtered options with columns: seq_idx, strike, time_to_expiry,
            rate_input, moneyness_input, vol_regime_input, mid_price (label).
        sequences: Array of shape (n_unique_dates, seq_len, 20). Each option's
            seq_idx column indexes into this array.
    """
    opts = opts_df.copy()

    # Standardize column names: lowercase, strip whitespace and brackets
    opts.columns = [c.lower().strip().replace(" ", "_").strip("[]") for c in opts.columns]

    # Map alternative column names to expected names
    col_aliases = {
        "date": ["date", "quote_date", "trade_date", "data_date"],
        "expiration": ["expiration", "expire_date", "expiry", "expiration_date"],
        "strike": ["strike", "strike_price"],
        "option_type": ["option_type", "type", "cp_flag", "call_put", "right"],
        "bid": ["bid", "best_bid", "bid_price", "c_bid"],
        "ask": ["ask", "best_offer", "ask_price", "offer", "c_ask"],
        "mid_price": ["mid_price", "mid", "price"],
        "underlying_price": ["underlying_price", "underlying_last", "stock_price",
                             "spot_price", "close", "underlying_close"],
        "implied_volatility": ["implied_volatility", "iv", "impl_volatility", "c_iv"],
    }

    for target, candidates in col_aliases.items():
        if target not in opts.columns:
            for alias in candidates:
                if alias in opts.columns:
                    opts.rename(columns={alias: target}, inplace=True)
                    break

    # Ensure date parsing
    for col in ["date", "expiration"]:
        if col in opts.columns:
            opts[col] = pd.to_datetime(opts[col])

    # Keep calls only — if option_type column exists, filter.
    # Kaggle SPY EOD format has C_/P_ columns per row (no option_type column),
    # so bid/ask already map to call columns via aliases above.
    if "option_type" in opts.columns:
        type_vals = opts["option_type"].astype(str).str.upper().str.strip()
        opts = opts[type_vals.isin(["C", "CALL"])].copy()

    # Compute derived features
    opts["time_to_expiry"] = (opts["expiration"] - opts["date"]).dt.days / 365.0
    opts["moneyness"] = opts["underlying_price"] / opts["strike"]
    if "mid_price" not in opts.columns:
        opts["mid_price"] = (opts["bid"] + opts["ask"]) / 2.0

    # Filter to interesting regime
    opts = opts[opts["time_to_expiry"].between(7 / 365, 90 / 365)].copy()
    opts = opts[opts["moneyness"].between(0.85, 1.15)].copy()
    opts = opts[opts["mid_price"] > 0.05].copy()

    # Scale the master features
    master_scaled_values = scaler.transform(master_df[FEATURE_COLS].values)

    # Build per-date sequences (vectorized — ~3500 unique dates vs millions of rows)
    trading_days = master_df.index
    date_to_iloc = {d: i for i, d in enumerate(trading_days)}

    date_seq_list = []        # list of (60, 20) arrays
    date_to_seq_idx = {}      # date -> index into date_seq_list
    date_spy_close = {}
    date_rate = {}
    date_vol_regime = {}

    for d in opts["date"].unique():
        d_ts = pd.Timestamp(d)
        if d_ts not in date_to_iloc:
            mask = trading_days <= d_ts
            if mask.sum() == 0:
                continue
            d_ts = trading_days[mask][-1]

        idx = date_to_iloc[d_ts]
        if idx < seq_len:
            continue

        date_to_seq_idx[d] = len(date_seq_list)
        date_seq_list.append(master_scaled_values[idx - seq_len: idx].astype(np.float32))
        t_minus_1 = master_df.iloc[idx - 1]
        date_spy_close[d] = t_minus_1["spy_close"]
        date_rate[d] = t_minus_1["treasury_2y"]
        date_vol_regime[d] = t_minus_1.get("vol_regime", 1)

    # Stack into a single array: (n_unique_dates, seq_len, 20)
    sequences_array = np.stack(date_seq_list) if date_seq_list else np.empty((0, seq_len, len(FEATURE_COLS)), dtype=np.float32)

    # Keep only options with valid dates
    valid_dates = set(date_to_seq_idx.keys())
    opts = opts[opts["date"].isin(valid_dates)].copy()

    # Store integer index into sequences_array (not the array itself)
    opts["seq_idx"] = opts["date"].map(date_to_seq_idx).astype(int)
    opts["moneyness_input"] = opts["date"].map(date_spy_close) / opts["strike"]
    opts["rate_input"] = opts["date"].map(date_rate)
    opts["vol_regime_input"] = opts["date"].map(date_vol_regime)

    opts = opts.reset_index(drop=True)

    return opts, sequences_array


def split_options(opts, cutoff="2020-01-01"):
    """Strict temporal train/test split for options data."""
    cutoff = pd.Timestamp(cutoff)
    opts_train = opts[opts["date"] < cutoff].copy().reset_index(drop=True)
    opts_test = opts[opts["date"] >= cutoff].copy().reset_index(drop=True)
    return opts_train, opts_test


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_full_pipeline(raw_dir=None, processed_dir=None):
    """
    Run the complete preprocessing pipeline end-to-end.

    Expects raw data files already present in raw_dir.

    Saves to processed_dir:
        - master_df.csv
        - scaler.pkl
        - train_sequences.npy
        - opts_train.pkl, opts_test.pkl
    """
    raw_dir = raw_dir or os.path.abspath(RAW_DIR)
    processed_dir = processed_dir or os.path.abspath(PROCESSED_DIR)
    os.makedirs(processed_dir, exist_ok=True)

    print("Loading raw data...")
    spy = pd.read_csv(os.path.join(raw_dir, "spy.csv"), index_col=0, parse_dates=True)
    vix = pd.read_csv(os.path.join(raw_dir, "vix.csv"), index_col=0, parse_dates=True)

    vvix = None
    vvix_path = os.path.join(raw_dir, "vvix.csv")
    if os.path.exists(vvix_path):
        vvix = pd.read_csv(vvix_path, index_col=0, parse_dates=True)

    hyg = pd.read_csv(os.path.join(raw_dir, "hyg.csv"), index_col=0, parse_dates=True)
    lqd = pd.read_csv(os.path.join(raw_dir, "lqd.csv"), index_col=0, parse_dates=True)
    dxy = pd.read_csv(os.path.join(raw_dir, "dxy.csv"), index_col=0, parse_dates=True)

    treasury_path = os.path.join(raw_dir, "treasury.csv")
    treasury = pd.read_csv(treasury_path, index_col=0, parse_dates=True) if os.path.exists(treasury_path) else None

    from src.data.download import load_cboe_files
    cboe = load_cboe_files(raw_dir)

    print("Building master DataFrame...")
    master = build_master_dataframe(spy, vix, vvix, hyg, lqd, dxy, treasury, cboe)
    master.to_csv(os.path.join(processed_dir, "master_df.csv"))

    print("Cleaning and splitting...")
    train_df, test_df = clean_and_split(master)
    print(f"  Train: {len(train_df)} days, Test: {len(test_df)} days")

    print("Fitting scaler on training data...")
    scaler = fit_scaler(train_df, os.path.join(processed_dir, "scaler.pkl"))

    print("Scaling and building sequences...")
    train_scaled = scale_data(train_df, scaler)
    test_scaled = scale_data(test_df, scaler)

    train_seqs = build_sequences(train_scaled)
    print(f"  Training sequences shape: {train_seqs.shape}")
    np.save(os.path.join(processed_dir, "train_sequences.npy"), train_seqs)
    np.save(os.path.join(processed_dir, "train_scaled.npy"), train_scaled)
    np.save(os.path.join(processed_dir, "test_scaled.npy"), test_scaled)

    # Save the index arrays for later use
    train_df.index.to_frame().to_csv(os.path.join(processed_dir, "train_dates.csv"))
    test_df.index.to_frame().to_csv(os.path.join(processed_dir, "test_dates.csv"))

    # Options preprocessing (if options data exists)
    opts_found = False
    from src.data.download import load_options_data
    try:
        opts_raw = load_options_data(raw_dir)
        print(f"Preprocessing options data ({len(opts_raw)} rows)...")
        opts, sequences_arr = preprocess_options(opts_raw, master, scaler)
        opts_train, opts_test = split_options(opts)
        print(f"  Options train: {len(opts_train)}, test: {len(opts_test)}")
        print(f"  Unique-date sequences: {sequences_arr.shape}")

        opts_train.to_pickle(os.path.join(processed_dir, "opts_train.pkl"))
        opts_test.to_pickle(os.path.join(processed_dir, "opts_test.pkl"))
        np.save(os.path.join(processed_dir, "opts_sequences.npy"), sequences_arr)
        opts_found = True
    except FileNotFoundError:
        pass

    if not opts_found:
        print("WARNING: No options CSV found. Skipping options preprocessing.")

    print("Preprocessing complete.")
    return master, scaler
