"""
Stage 5: finite-difference Greeks

Computes Delta, Gamma, Vega, Theta, Rho by nudging each input slightly and 
re-pricing, using the Binomial Tree (deterministic) or Monte Carlo (noisy)
pricers.
Cross-checked against black_scholes.py's closed-form Greeks -- 
if these two independently-computed sets of numbers agree, that's
real evidenve that the whole pricing engine is implemented correctly,
not just the Black-Scholes formula in isolation 

"""
import numpy as np 

from black_scholes import black_scholes_price, black_scholes_greeks
from binomial_tree import binomial_tree_price
from monte_carlo import monte_carlo_price

def finite_diff_greeks(pricer_fn, S, K, T, r, sigma, option_type = "call",
                       eps_S = 0.5, eps_sigma = 0.001, eps_T = 1 / 365, eps_r = 0.0001):
    """
    Compute all 5 greeks via finite differences, using whichever pricing 
    function is passed (binomial tree / monte carlo)
    `pricer_fn` must accept (S, K, T, r, sigma, option_type) and return 
    a single float price.

    Central differences are used throughout
    Since they're more accurate than a one-sided bump for the same 
    step size
    """

    # Delta: sensitivity to spot price 
    price_up = pricer_fn(S + eps_S, K, T, r, sigma, option_type)
    price_down = pricer_fn(S - eps_S, K, T, r, sigma, option_type)
    delta = (price_up - price_down) / (2 * eps_S)

    # Gamma: sensitivity of Delta to spot price
    price_center = pricer_fn(S, K, T, r, sigma, option_type)
    gamma = (price_up - 2 * price_center + price_down) / (eps_S ** 2)

    # Vega: sensitivity to volatility 
    price_vol_up = pricer_fn(S, K, T, r, sigma + eps_sigma, option_type)
    price_vol_down = pricer_fn(S, K, T, r, sigma - eps_sigma, option_type)

    vega = (price_vol_up - price_vol_down) / (2 * eps_sigma)

    # Theta: sensitivity to time (dPrice / dT)
    price_T_up = pricer_fn(S, K, T, eps_T, r, sigma, option_type)
    price_T_down = pricer_fn(S, K, T - eps_T, r, sigma, option_type)
    theta = -(price_T_up - price_T_down) / (2 * eps_T)

    # Rho: sensitivity to interest rate 
    price_r_up = pricer_fn(S, K, T, r + eps_r, sigma, option_type)
    price_r_down = pricer_fn(S, K, T, r - eps_r, sigma, option_type)
    rho = (price_r_up - price_r_down) / (2 * eps_r)

    return{
        "delta": delta,
        "gamma": gamma,
        "vega" : vega, 
        "theta": theta,
        "rho" : rho
    }

def _binomial_wrapper(steps = 500):
    """
    Returns a pricer_fn matching the (S, K, T, r, sigma, option_type)
    signature that finite_diff_greeks expects, with a 
    fixed number of steps 
    """
    def wrapped(S, K, T, r, sigma, option_type):
        return binomial_tree_price(S, K, T, r, sigma, option_type, steps=steps)
    return wrapped 

def _monte_carlo_wrapper(n_sims = 200_000, seed = 42):
    def wrapped(S, K, T, r, sigma, option_type):
        result = monte_carlo_price(S, K, T, r, sigma, option_type, n_sims = n_sims, seed = seed)
        return result["price"]
    return wrapped 
