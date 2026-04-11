"""Black-Scholes closed-form European call option pricing (baseline only, never trained)."""

import numpy as np
from scipy.stats import norm


def black_scholes_call(S, K, T, r, sigma):
    """
    Compute Black-Scholes European call option price.

    Parameters
    ----------
    S : float or array
        Current underlying price (SPY close at t-1).
    K : float or array
        Strike price.
    T : float or array
        Time to expiry in years.
    r : float or array
        Risk-free rate (treasury_2y at t-1).
    sigma : float or array
        Volatility input (rv_21d at t-1).

    Returns
    -------
    price : float or array
        Theoretical call option price.
    """
    S = np.asarray(S, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Guard against zero/negative vol or time
    sigma = np.maximum(sigma, 1e-8)
    T = np.maximum(T, 1e-8)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return price
