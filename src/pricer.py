"""
Stage 4: Unified Pricer Interface 

Wraps black-scholes, monte carlo and binomial tree into 
one consistent function signature 
so callers (comparison table, tests and the dashboard)
are abstracted from the internals of any of the pricers
Interfaces just send a request on which method they want

"""

from black_scholes import black_scholes_price
from monte_carlo import monte_carlo_price
from binomial_tree import binomial_tree_price

def price(method, S, K, T, r, sigma, option_type = "call", **kwargs):
    """
    Price a European option using the specified method
    
    Args:
        method: "bs" : Black-Scholes; "mc": Monte Carlo; "bt": Binomial Tree
        S, K, T, r, sigma, option_type: standard option pricing inputs 
        **kwargs: method-specific extras, e.g: n_sims/antithetic seeds for 
                  "mc"
    
    Returns:
        float: the option price (Monte Carlo's confidence interval)

    """
    if method == "bs":
        return black_scholes_price(S, K, T, r, sigma, option_type)
    elif method == "mc":
        result = monte_carlo_price(S, K, T, r, sigma, option_type, **kwargs)
        return result 
    elif method == "bt":
        return binomial_tree_price(S, K, T, r, sigma, option_type, **kwargs)

    else:
        raise ValueError(f"Unknown method '{method}'. Use 'bs', 'mc' or 'bt'. ")

def compare_methods(S, K, T, r, sigma, option_type = "call", n_sims= 100_000, steps = 200, seed = 42):
    """
    Price the same option all three ways and return a comparison dict 
    """       
    return {
        "black_scholes": price("bs", S, K, T, r, sigma, option_type),
        "monte_carlo": price("mc", S, K, T, r, sigma, option_type, n_sims = n_sims, seed = seed),
        "binomial_tree": price("bt", S, K, T, r, sigma, option_type, steps = steps),
    }

def print_comparison_table(S, K, T, r, sigma, n_sims = 100_000, steps = 200, seed = 42):
    """
    Print a side-by-side call/put comparison table across all 3 methods
    """
    call_results = compare_methods(S, K, T, r, sigma, "call", "n_sims", steps, seed)
    put_results = compare_methods(S, K, T, r, sigma, "put", "n_sims", steps, seed)

    print(f"{'Method:<15'} {'Call Price:'>12} {'Put Price:>12'}")
    print("-" * 41)
    for method_key, label in [("black_scholes", "Black-Scholes"), 
                              ("monte_carlo", "Monte Carlo"),
                              ("binomial_tree", "Binomial Tree")]:
        print(f"{label:<15} {call_results[method_key]:>12.4f} {put_results[method_key]:>12.4f}")

if __name__ == "__main__":
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    print(f"Pricing a European option: S={S}, K={K}, T={T}, r={r}, sigma={sigma}\n")
    print_comparison_table(S, K, T, r, sigma)
    print("\n--- Same call via the unified interface, three ways ---")
    print(f"price('bs', ...)       = {price('bs', S, K, T, r, sigma, 'call'):.4f}")
    print(f"price('mc', ...)       = {price('mc', S, K, T, r, sigma, 'call', n_sims=100_000, seed=42):.4f}")
    print(f"price('binomial', ...) = {price('binomial', S, K, T, r, sigma, 'call', steps=200):.4f}")