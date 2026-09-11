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
        