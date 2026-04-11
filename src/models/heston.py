"""Heston stochastic volatility Monte Carlo simulator for European call options."""

import numpy as np


# Default Heston parameters calibrated to SPY dynamics
HESTON_DEFAULTS = dict(
    mu=0.05,      # annual drift
    kappa=2.0,    # mean-reversion speed of variance
    theta=0.04,   # long-run variance (≈20% annual vol)
    xi=0.3,       # vol of vol
    rho=-0.7,     # price-vol correlation (leverage effect)
    v0=0.04,      # initial variance
)


def simulate_heston(S0, K, T, r, n_paths=10000, n_steps=252,
                    mu=None, kappa=None, theta=None, xi=None, rho=None, v0=None,
                    return_paths=False, seed=None):
    """
    Simulate Heston stochastic volatility paths and price a European call.

    Parameters
    ----------
    S0 : float
        Initial underlying price.
    K : float
        Strike price.
    T : float
        Time to expiry in years.
    r : float
        Risk-free rate.
    n_paths : int
        Number of Monte Carlo paths.
    n_steps : int
        Number of time steps per path.
    mu, kappa, theta, xi, rho, v0 : float or None
        Heston model parameters. None falls back to HESTON_DEFAULTS.
    return_paths : bool
        If True, also return the full price path array.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    price : float
        Monte Carlo estimated European call price = e^{-rT} * E[max(S_T - K, 0)].
    paths : np.ndarray, shape (n_paths, n_steps+1)
        Only returned if return_paths=True.
    """
    # Fill defaults
    p = HESTON_DEFAULTS.copy()
    for name, val in [("mu", mu), ("kappa", kappa), ("theta", theta),
                      ("xi", xi), ("rho", rho), ("v0", v0)]:
        if val is not None:
            p[name] = val

    if seed is not None:
        np.random.seed(seed)

    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)

    S = np.zeros((n_paths, n_steps + 1))
    v = np.zeros((n_paths, n_steps + 1))
    S[:, 0] = S0
    v[:, 0] = p["v0"]

    for t in range(n_steps):
        Z1 = np.random.normal(size=n_paths)
        Z2 = p["rho"] * Z1 + np.sqrt(1 - p["rho"] ** 2) * np.random.normal(size=n_paths)

        # Variance process (Euler-Maruyama with floor at 0)
        v_cur = np.maximum(v[:, t], 0)
        v[:, t + 1] = np.maximum(
            v[:, t] + p["kappa"] * (p["theta"] - v[:, t]) * dt
            + p["xi"] * np.sqrt(v_cur) * sqrt_dt * Z2,
            0,
        )

        # Price process
        S[:, t + 1] = S[:, t] * np.exp(
            (r - 0.5 * v_cur) * dt + np.sqrt(v_cur * dt) * Z1
        )

    # Discounted expected payoff
    payoffs = np.maximum(S[:, -1] - K, 0)
    price = np.exp(-r * T) * np.mean(payoffs)

    if return_paths:
        return price, S
    return price


def generate_heston_training_data(n_samples, S0_range=(300, 500), K_range=(280, 520),
                                  T_range=(7 / 365, 90 / 365), r_range=(0.01, 0.05),
                                  n_paths=10000, n_steps=252, seed=42):
    """
    Generate labeled training data for the Transformer pricing network using Heston paths.

    Each sample consists of randomly drawn contract parameters and the Monte Carlo
    price as the label.

    Parameters
    ----------
    n_samples : int
        Number of (contract, label) pairs to generate.
    S0_range, K_range, T_range, r_range : tuple
        Uniform sampling ranges for contract parameters.
    n_paths : int
        Monte Carlo paths per sample.
    n_steps : int
        Time steps per path.
    seed : int
        Base random seed.

    Returns
    -------
    contracts : np.ndarray, shape (n_samples, 4)
        Columns: [S0, K, T, r]
    labels : np.ndarray, shape (n_samples,)
        Discounted expected payoffs (Heston Monte Carlo prices).
    """
    rng = np.random.RandomState(seed)

    contracts = np.zeros((n_samples, 4))
    labels = np.zeros(n_samples)

    for i in range(n_samples):
        S0 = rng.uniform(*S0_range)
        K = rng.uniform(*K_range)
        T = rng.uniform(*T_range)
        r = rng.uniform(*r_range)

        contracts[i] = [S0, K, T, r]
        labels[i] = simulate_heston(S0, K, T, r, n_paths=n_paths, n_steps=n_steps,
                                    seed=seed + i)

    return contracts, labels
