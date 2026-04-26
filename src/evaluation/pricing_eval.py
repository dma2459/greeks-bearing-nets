"""
Pricing evaluation: MAE, MSE, MAPE computed overall and broken down by
volatility regime, moneyness bucket, and time-to-expiry bucket.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from src.data.dataset import OptionsDataset
from src.models.black_scholes import black_scholes_call


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred):
    """Compute MAE, MSE, MAPE."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mae = np.mean(np.abs(y_pred - y_true))
    mse = np.mean((y_pred - y_true) ** 2)
    # MAPE: guard against near-zero labels
    mask = np.abs(y_true) > 1e-6
    mape = np.mean(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask])) * 100 if mask.any() else np.nan
    return {"MAE": mae, "MSE": mse, "MAPE": mape}


# ---------------------------------------------------------------------------
# Bucketing helpers
# ---------------------------------------------------------------------------

def assign_vol_regime(vix_values):
    """Assign volatility regime labels: 0=low, 1=medium, 2=high."""
    regimes = np.ones_like(vix_values, dtype=int)
    regimes[vix_values < 15] = 0
    regimes[vix_values > 25] = 2
    return regimes


def assign_moneyness_bucket(moneyness):
    """OTM (<0.97), ATM (0.97-1.03), ITM (>1.03)."""
    buckets = np.full(len(moneyness), "ATM", dtype=object)
    buckets[moneyness < 0.97] = "OTM"
    buckets[moneyness > 1.03] = "ITM"
    return buckets


def assign_expiry_bucket(tte_days):
    """Short (7-30d), Medium (30-60d), Long (60-90d)."""
    tte_days = np.asarray(tte_days)
    buckets = np.full(len(tte_days), "medium", dtype=object)
    buckets[tte_days <= 30] = "short"
    buckets[tte_days > 60] = "long"
    return buckets


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def predict_transformer(model, opts_df, sequences=None, batch_size=512, device=None):
    """
    Run inference on preprocessed options DataFrame using a Transformer.

    Parameters
    ----------
    sequences : np.ndarray or None
        Shape (n_unique_dates, seq_len, 20). Required — each option's seq_idx
        column indexes into this array.

    Returns np.array of predictions aligned with opts_df rows.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else
                              "mps" if torch.backends.mps.is_available() else "cpu")
    if sequences is None:
        raise ValueError("sequences array is required for predict_transformer")

    dataset = OptionsDataset(opts_df, sequences)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Figure out which inverse transform to apply. Run 3 uses target_mode;
    # Run 2 checkpoints only have log_target — honor both for compatibility.
    target_mode = getattr(model, "target_mode", None)
    if target_mode is None:
        target_mode = "log" if getattr(model, "log_target", False) else "raw"

    model = model.to(device).eval()
    preds = []
    intrinsics = []  # only needed when target_mode == "time_value"
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            out = model(inputs)
            preds.append(out.cpu().numpy())
            if target_mode == "time_value":
                # Contract block is 8-wide; K at channel -8, moneyness at -5
                # (see src/data/dataset.py:augment_contract_features).
                K = inputs[:, 0, -8].cpu().numpy()
                moneyness = inputs[:, 0, -5].cpu().numpy()
                intrinsics.append(np.maximum((moneyness - 1.0) * K, 0.0))

    raw = np.concatenate(preds, axis=0).flatten()
    if target_mode == "time_value":
        intrinsic = np.concatenate(intrinsics, axis=0)
        raw = np.expm1(raw) + intrinsic
    elif target_mode == "log":
        raw = np.expm1(raw)
    # Clamp to non-negative (the head no longer has a final ReLU — that was
    # causing A3/A6 to collapse to the dead-ReLU "predict 0 always" minimum).
    return np.maximum(raw, 0.0)


def modify_sequences_for_ablation(sequences, ablation_id):
    """
    Modify sequences array for a specific ablation study.

    Thin wrapper around src.data.preprocess.prepare_ablation_sequences. Expects
    raw (N, 60, 20) input and returns the properly-prepped sequences for the
    ablation (feature drops + seq_len slicing as defined by Run 3 baseline).
    """
    from src.data.preprocess import prepare_ablation_sequences
    return prepare_ablation_sequences(sequences, ablation_id)


def predict_black_scholes(opts_df, master_df=None):
    """
    Compute Black-Scholes prices for each option using day-t (EOD) values.

    S, r, and sigma are all taken from trading day t — the contract's quote
    date — matching the EOD convention used by preprocess_options. This puts
    BS on equal footing with the Transformer, which also sees day-t market
    state in its input sequence.

    Parameters
    ----------
    opts_df : pd.DataFrame
        Must have: strike, time_to_expiry, rate_input, moneyness_input.
        If rv_21d_input is present, it is used directly for sigma.
    master_df : pd.DataFrame or None
        Fallback source for sigma when rv_21d_input/rv_21d are absent.

    Returns
    -------
    np.ndarray
        BS prices.
    """
    S = opts_df["moneyness_input"].values * opts_df["strike"].values
    K = opts_df["strike"].values
    T = opts_df["time_to_expiry"].values
    r = opts_df["rate_input"].values

    # Prefer the day-t rv_21d populated by preprocess_options.
    if "rv_21d_input" in opts_df.columns:
        sigma = opts_df["rv_21d_input"].values.astype(np.float64)
    elif "rv_21d" in opts_df.columns:
        sigma = opts_df["rv_21d"].values.astype(np.float64)
    elif master_df is not None and "rv_21d" in master_df.columns:
        # Fallback: look up day-t rv_21d (last trading day <= quote date).
        sigmas = []
        trading_days = master_df.index
        for _, row in opts_df.iterrows():
            d = row["date"]
            mask = trading_days <= d
            if mask.any():
                idx = trading_days[mask][-1]  # day-t
                sigmas.append(master_df.loc[idx, "rv_21d"])
            else:
                sigmas.append(0.2)
        sigma = np.array(sigmas, dtype=np.float64)
    else:
        sigma = np.full(len(opts_df), 0.2)

    # Any NaNs (e.g. early-history rv_21d warmup) fall back to a reasonable default.
    sigma = np.where(np.isnan(sigma), 0.2, sigma)
    # Guard against zero sigma
    sigma = np.maximum(sigma, 0.01)

    return black_scholes_call(S, K, T, r, sigma)


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------

def evaluate_all_models(opts_test, models_dict, sequences=None, master_df=None,
                        device=None, figures_dir=None, tables_dir=None):
    """
    Evaluate all models on the test options set.

    Parameters
    ----------
    opts_test : pd.DataFrame
        Preprocessed test options.
    models_dict : dict
        {name: model} where model is a TransformerPricingNetwork or "BS" for Black-Scholes.
    sequences : np.ndarray
        Shape (n_unique_dates, seq_len, 20). Required for Transformer models.
    master_df : pd.DataFrame or None
        Needed for BS rv_21d lookup.
    device : torch.device or None
    figures_dir, tables_dir : str or None

    Returns
    -------
    pd.DataFrame
        Results table with metrics by model and breakdown.
    """
    if figures_dir is None:
        figures_dir = os.path.join("results", "figures")
    if tables_dir is None:
        tables_dir = os.path.join("results", "tables")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    y_true = opts_test["mid_price"].values

    # ── Get predictions from all models ──
    predictions = {}
    for name, model in models_dict.items():
        print(f"Predicting with {name}...")
        if name.lower() == "black-scholes" or model == "BS":
            predictions[name] = predict_black_scholes(opts_test, master_df)
        else:
            predictions[name] = predict_transformer(model, opts_test, sequences=sequences, device=device)

    # ── Overall metrics ──
    overall_rows = []
    for name, y_pred in predictions.items():
        m = compute_metrics(y_true, y_pred)
        m["Model"] = name
        overall_rows.append(m)
    overall_df = pd.DataFrame(overall_rows).set_index("Model")
    print("\n=== Overall Metrics ===")
    print(overall_df.to_string())
    overall_df.to_csv(os.path.join(tables_dir, "overall_metrics.csv"))

    # ── Breakdown by volatility regime ──
    if "vol_regime_input" in opts_test.columns:
        regime_labels = opts_test["vol_regime_input"].values
    elif "vix" in opts_test.columns:
        regime_labels = assign_vol_regime(opts_test["vix"].values)
    else:
        regime_labels = np.ones(len(opts_test))

    regime_names = {0: "Low (VIX<15)", 1: "Medium (15-25)", 2: "High (VIX>25)"}
    regime_rows = []
    for regime_val in [0, 1, 2]:
        mask = regime_labels == regime_val
        if not mask.any():
            continue
        for name, y_pred in predictions.items():
            m = compute_metrics(y_true[mask], y_pred[mask])
            m["Model"] = name
            m["Regime"] = regime_names.get(regime_val, str(regime_val))
            regime_rows.append(m)
    regime_df = pd.DataFrame(regime_rows)
    if len(regime_df):
        print("\n=== By Volatility Regime — MAE ($) ===")
        print(regime_df.pivot(index="Model", columns="Regime", values="MAE").round(3).to_string())
        print("\n=== By Volatility Regime — MAPE (%) ===")
        print(regime_df.pivot(index="Model", columns="Regime", values="MAPE").round(2).to_string())
        regime_df.to_csv(os.path.join(tables_dir, "regime_metrics.csv"), index=False)

    # ── Breakdown by moneyness ──
    moneyness = opts_test["moneyness_input"].values if "moneyness_input" in opts_test.columns \
                else opts_test["moneyness"].values
    money_buckets = assign_moneyness_bucket(moneyness)
    money_rows = []
    for bucket in ["OTM", "ATM", "ITM"]:
        mask = money_buckets == bucket
        if not mask.any():
            continue
        for name, y_pred in predictions.items():
            m = compute_metrics(y_true[mask], y_pred[mask])
            m["Model"] = name
            m["Moneyness"] = bucket
            money_rows.append(m)
    money_df = pd.DataFrame(money_rows)
    if len(money_df):
        print("\n=== By Moneyness — MAE ($) ===")
        print(money_df.pivot(index="Model", columns="Moneyness", values="MAE").round(3).to_string())
        print("\n=== By Moneyness — MAPE (%) ===")
        print(money_df.pivot(index="Model", columns="Moneyness", values="MAPE").round(2).to_string())
        money_df.to_csv(os.path.join(tables_dir, "moneyness_metrics.csv"), index=False)

    # ── Breakdown by time to expiry ──
    tte_days = opts_test["time_to_expiry"].values * 365
    expiry_buckets = assign_expiry_bucket(tte_days)
    expiry_rows = []
    for bucket in ["short", "medium", "long"]:
        mask = expiry_buckets == bucket
        if not mask.any():
            continue
        for name, y_pred in predictions.items():
            m = compute_metrics(y_true[mask], y_pred[mask])
            m["Model"] = name
            m["Expiry"] = bucket
            expiry_rows.append(m)
    expiry_df = pd.DataFrame(expiry_rows)
    if len(expiry_df):
        print("\n=== By Time to Expiry — MAE ($) ===")
        print(expiry_df.pivot(index="Model", columns="Expiry", values="MAE").round(3).to_string())
        print("\n=== By Time to Expiry — MAPE (%) ===")
        print(expiry_df.pivot(index="Model", columns="Expiry", values="MAPE").round(2).to_string())
        expiry_df.to_csv(os.path.join(tables_dir, "expiry_metrics.csv"), index=False)

    # ── Comparison bar chart ──
    _plot_comparison(overall_df, figures_dir)

    return overall_df


def _plot_comparison(overall_df, figures_dir):
    """Bar chart of MAE across models."""
    fig, ax = plt.subplots(figsize=(10, 6))
    models = overall_df.index.tolist()
    maes = overall_df["MAE"].values
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
    bars = ax.bar(models, maes, color=colors)

    for bar, mae in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{mae:.4f}", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("MAE ($)")
    ax.set_title("Model Comparison: Mean Absolute Error on Test Set (2020-2023)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(figures_dir, "model_comparison_mae.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved comparison chart to {path}")
    plt.show()


def evaluate_ablations(opts_test, ablation_models, baseline_model, sequences=None,
                       master_df=None, device=None, tables_dir=None):
    """
    Evaluate ablation models vs baseline and report a results table.

    Parameters
    ----------
    opts_test : pd.DataFrame
    ablation_models : dict
        {ablation_id: model}
    baseline_model : TransformerPricingNetwork
        Experiment C baseline.
    sequences : np.ndarray
        Shape (n_unique_dates, seq_len, 20). Required for Transformer models.
    master_df : pd.DataFrame or None
    device : torch.device or None
    tables_dir : str or None

    Returns
    -------
    pd.DataFrame
    """
    if tables_dir is None:
        tables_dir = os.path.join("results", "tables")
    os.makedirs(tables_dir, exist_ok=True)

    y_true = opts_test["mid_price"].values

    # Baseline predictions
    baseline_preds = predict_transformer(baseline_model, opts_test,
                                         sequences=sequences, device=device)
    baseline_metrics = compute_metrics(y_true, baseline_preds)

    rows = [{"Ablation": "Baseline (C)", **baseline_metrics}]

    for abl_id, model in ablation_models.items():
        preds = predict_transformer(model, opts_test, sequences=sequences, device=device)
        m = compute_metrics(y_true, preds)
        m["Ablation"] = abl_id
        m["MAE_diff"] = m["MAE"] - baseline_metrics["MAE"]
        rows.append(m)

    df = pd.DataFrame(rows).set_index("Ablation")
    print("\n=== Ablation Results ===")
    print(df.to_string())
    df.to_csv(os.path.join(tables_dir, "ablation_results.csv"))
    return df
