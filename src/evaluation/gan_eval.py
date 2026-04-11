"""
GAN quality evaluation: distribution matching, volatility clustering, MMD score.

Run all three checks before proceeding to the pricing network.
"""

import numpy as np
import matplotlib.pyplot as plt
import os


# ---------------------------------------------------------------------------
# Check 1: Distribution matching — histogram of log returns, kurtosis
# ---------------------------------------------------------------------------

def plot_return_distribution(real_sequences, fake_sequences, feature_idx=0,
                             feature_name="log_return", save_path=None):
    """
    Plot overlaid histograms of a feature from real vs synthetic sequences.

    Parameters
    ----------
    real_sequences : np.ndarray, shape (N, seq_len, n_features)
    fake_sequences : np.ndarray, shape (M, seq_len, n_features)
    feature_idx : int
        Column index of the feature to plot.
    feature_name : str
    save_path : str or None
    """
    real_vals = real_sequences[:, :, feature_idx].flatten()
    fake_vals = fake_sequences[:, :, feature_idx].flatten()

    real_kurt = float(np.mean((real_vals - real_vals.mean()) ** 4) / real_vals.var() ** 2)
    fake_kurt = float(np.mean((fake_vals - fake_vals.mean()) ** 4) / fake_vals.var() ** 2)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(real_vals, bins=100, alpha=0.5, density=True, label=f"Real (kurtosis={real_kurt:.2f})")
    ax.hist(fake_vals, bins=100, alpha=0.5, density=True, label=f"Synthetic (kurtosis={fake_kurt:.2f})")
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_title(f"Distribution Matching: {feature_name}")
    ax.set_xlabel(feature_name)
    ax.set_ylabel("Density")
    ax.legend()
    ax.text(0.02, 0.95, f"Gaussian kurtosis = 3.0", transform=ax.transAxes,
            fontsize=9, verticalalignment="top", color="gray")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved distribution plot to {save_path}")

    plt.show()
    return real_kurt, fake_kurt


# ---------------------------------------------------------------------------
# Check 2: Volatility clustering — autocorrelation of squared returns
# ---------------------------------------------------------------------------

def compute_autocorrelation(series, max_lag=30):
    """Compute autocorrelation of a 1D series up to max_lag."""
    series = series - series.mean()
    n = len(series)
    var = np.sum(series ** 2)
    acf = np.zeros(max_lag + 1)
    for lag in range(max_lag + 1):
        acf[lag] = np.sum(series[: n - lag] * series[lag:]) / var
    return acf


def plot_volatility_clustering(real_sequences, fake_sequences, feature_idx=0,
                               max_lag=30, save_path=None):
    """
    Plot autocorrelation of squared log returns for real vs synthetic sequences.

    Real markets show slow hyperbolic decay (ARCH effects).

    Parameters
    ----------
    real_sequences, fake_sequences : np.ndarray, shape (N, seq_len, n_features)
    feature_idx : int
        Index of log_return feature.
    max_lag : int
    save_path : str or None
    """
    # Pool all sequences into long series and compute squared returns
    real_returns = real_sequences[:, :, feature_idx].flatten()
    fake_returns = fake_sequences[:, :, feature_idx].flatten()

    real_sq = real_returns ** 2
    fake_sq = fake_returns ** 2

    real_acf = compute_autocorrelation(real_sq, max_lag)
    fake_acf = compute_autocorrelation(fake_sq, max_lag)

    fig, ax = plt.subplots(figsize=(10, 6))
    lags = np.arange(max_lag + 1)
    ax.plot(lags, real_acf, "o-", label="Real", markersize=4)
    ax.plot(lags, fake_acf, "s-", label="Synthetic", markersize=4)
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Volatility Clustering: Autocorrelation of Squared Returns")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved vol clustering plot to {save_path}")

    plt.show()
    return real_acf, fake_acf


# ---------------------------------------------------------------------------
# Check 3: Maximum Mean Discrepancy (MMD)
# ---------------------------------------------------------------------------

def _rbf_kernel(X, Y, sigma=1.0):
    """Compute RBF (Gaussian) kernel between flattened sequence matrices."""
    XX = np.sum(X ** 2, axis=1, keepdims=True)
    YY = np.sum(Y ** 2, axis=1, keepdims=True)
    dists = XX + YY.T - 2 * X @ Y.T
    return np.exp(-dists / (2 * sigma ** 2))


def compute_mmd(X, Y, n_samples=1000, sigma=1.0, seed=42):
    """
    Compute Maximum Mean Discrepancy between two sets of sequences.

    Parameters
    ----------
    X, Y : np.ndarray, shape (N, seq_len, n_features)
    n_samples : int
        Subsample size for computational tractability.
    sigma : float
        RBF kernel bandwidth.
    seed : int

    Returns
    -------
    float
        MMD^2 estimate.
    """
    rng = np.random.RandomState(seed)

    # Subsample
    if len(X) > n_samples:
        idx = rng.choice(len(X), n_samples, replace=False)
        X = X[idx]
    if len(Y) > n_samples:
        idx = rng.choice(len(Y), n_samples, replace=False)
        Y = Y[idx]

    # Flatten sequences: (N, seq_len * n_features)
    X_flat = X.reshape(len(X), -1)
    Y_flat = Y.reshape(len(Y), -1)

    K_XX = _rbf_kernel(X_flat, X_flat, sigma)
    K_YY = _rbf_kernel(Y_flat, Y_flat, sigma)
    K_XY = _rbf_kernel(X_flat, Y_flat, sigma)

    n = len(X_flat)
    m = len(Y_flat)

    # Unbiased MMD^2 estimator
    mmd2 = (np.sum(K_XX) - np.trace(K_XX)) / (n * (n - 1)) \
         + (np.sum(K_YY) - np.trace(K_YY)) / (m * (m - 1)) \
         - 2 * np.sum(K_XY) / (n * m)

    return float(mmd2)


def run_mmd_check(real_train, real_val, fake_sequences, n_samples=1000, sigma=1.0,
                  save_path=None):
    """
    Compare baseline MMD (real train vs real val) with GAN MMD (real train vs fake).

    If gan_MMD is within 2x of baseline_MMD, the GAN has learned the distribution
    reasonably well.

    Parameters
    ----------
    real_train : np.ndarray, shape (N1, seq_len, n_features)
    real_val : np.ndarray, shape (N2, seq_len, n_features)
    fake_sequences : np.ndarray, shape (M, seq_len, n_features)

    Returns
    -------
    dict with baseline_mmd, gan_mmd, ratio, passed (bool)
    """
    print("Computing baseline MMD (real_train vs real_val)...")
    baseline_mmd = compute_mmd(real_train, real_val, n_samples=n_samples, sigma=sigma)
    print(f"  Baseline MMD: {baseline_mmd:.6f}")

    print("Computing GAN MMD (real_train vs synthetic)...")
    gan_mmd = compute_mmd(real_train, fake_sequences, n_samples=n_samples, sigma=sigma)
    print(f"  GAN MMD:      {gan_mmd:.6f}")

    ratio = gan_mmd / max(baseline_mmd, 1e-10)
    passed = ratio < 2.0

    print(f"  Ratio (GAN/baseline): {ratio:.2f}")
    print(f"  {'PASS' if passed else 'FAIL'}: {'within' if passed else 'exceeds'} 2x threshold")

    result = {
        "baseline_mmd": baseline_mmd,
        "gan_mmd": gan_mmd,
        "ratio": ratio,
        "passed": passed,
    }

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            for k, v in result.items():
                f.write(f"{k}: {v}\n")

    return result


# ---------------------------------------------------------------------------
# Run all quality checks
# ---------------------------------------------------------------------------

def run_all_quality_checks(real_train, real_val, fake_sequences, figures_dir=None):
    """
    Run all three GAN quality checks.

    Parameters
    ----------
    real_train : np.ndarray, shape (N1, seq_len, n_features)
    real_val : np.ndarray, shape (N2, seq_len, n_features)
    fake_sequences : np.ndarray, shape (M, seq_len, n_features)
    figures_dir : str or None

    Returns
    -------
    dict with all results.
    """
    if figures_dir is None:
        figures_dir = os.path.join("results", "figures")

    print("\n" + "=" * 60)
    print("GAN Quality Check 1: Distribution Matching")
    print("=" * 60)
    real_kurt, fake_kurt = plot_return_distribution(
        real_train, fake_sequences,
        save_path=os.path.join(figures_dir, "gan_distribution.png"),
    )

    print("\n" + "=" * 60)
    print("GAN Quality Check 2: Volatility Clustering")
    print("=" * 60)
    real_acf, fake_acf = plot_volatility_clustering(
        real_train, fake_sequences,
        save_path=os.path.join(figures_dir, "gan_vol_clustering.png"),
    )

    print("\n" + "=" * 60)
    print("GAN Quality Check 3: MMD Score")
    print("=" * 60)
    mmd_result = run_mmd_check(
        real_train, real_val, fake_sequences,
        save_path=os.path.join(figures_dir, "gan_mmd_results.txt"),
    )

    return {
        "real_kurtosis": real_kurt,
        "fake_kurtosis": fake_kurt,
        "real_acf": real_acf,
        "fake_acf": fake_acf,
        **mmd_result,
    }
