"""
Stage 2: Monte Carlo option pricer 

Prices European options by simulating many random future stock paths under 
the risk-neutral measure (same GBM assumption as Black-Scholes), averaging
the discounted payoffs, and reporting a confidence interval on that 
estimate 

This should converge to black_scholes.py's output as n_sims grows
that convergence is the actual point of this module, not just "get a number"

"""

import numpy as np
from black_scholes import black_scholes_price

def monte_carlo_price(S, K, T, r, sigma, option_type="call", n_sims=100_000, antithetic=True, seed= None):
    """
    Price a European option via Monte Carlo Simulation 

    Args:
        antithetic: if True, use antithetic variaties for variance reduction
                    (pair each random draw Z with -Z)
        
        seed: optional RNG seed, for reproducibility
    
    Returns:
        dict with keys: price, std_error, ci_lower, ci_upper, n_sims
    """
    rng = np.random.default_rng(seed)

    if antithetic:
        # Draw half as many independent Z's, then mirror with -Z

        half = n_sims // 2
        Z = rng.standard_normal(half)
        Z = np.concatenate([Z, -Z])
    else:
        Z = rng.standard_normal(n_sims)

    # Simulate terminal stock prices under the risk-neutral GBM process 
    
    S_T = S * np.exp((r - 0.5 * sigma **2) * T + sigma * np.sqrt(T) * Z)

    if option_type == "call":
        payoffs = np.maximum(S_T - K, 0)
    elif option_type == "put":
        payoffs = np.maximum(K - S_T, 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    discounted_payoffs = np.exp(-r * T) * payoffs

    price = discounted_payoffs.mean()

    std_error = discounted_payoffs.std(ddof=1) / np.sqrt(len(discounted_payoffs))

    ci_lower = price - 1.96 * std_error
    ci_upper = price + 1.96 * std_error

    return {
        "price": price,
        "std_error": std_error, 
        "ci_lower": ci_lower,
        "ci_upper": ci_upper, 
        "n_sims": n_sims,
    }

if __name__ == "__main__":
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2

    bs_call = black_scholes_price(S, K, T, r, sigma, "call")
    bs_put = black_scholes_price(S, K, T, r, sigma, "put")

    print(f"Black-Scholes call price (ground truth): {bs_call:.4f}")
    print(f"Black Scholes put price (ground truth): {bs_put:.4f}")

    for n in [1_000, 10_000, 100_000, 1_000_000]:
        result = monte_carlo_price(S, K, T, r, sigma, "call", n_sims=n, antithetic=True, seed = 42)

        print(
            f"n_sims={n:>9,} "
            f"MC call price: {result['price']:.4f}"
            f"95% CI:[{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]"
            f"error vs BS:{abs(result['price'] - bs_call):.4f}"
        )

        print()

        # Check that antithetic variates actually help 
        plain = monte_carlo_price(S, K, T, r, sigma, "call", n_sims = 100_000, antithetic = False, seed = 42)
        anti = monte_carlo_price(S, K, T, r, sigma, "call", n_sims = 100_000, antithetic = True, seed = 42)

        print(f"Std error WITHOUT antithetic variates: {plain['std_error']:.5f}")
        print(f"Std error WITH antithetic variates:    {anti['std_error']:.5f}")

        # Check: put ptice should also converge to Black-Scholes
        mc_put = monte_carlo_price(S, K, T, r, sigma, "put", n_sims = 1_000_000, antithetic=True, seed = 42)

        print(f"\nMC put price (1M sims): {mc_put['price']:.4f} "
          f"vs Black-Scholes put: {bs_put:.4f}")