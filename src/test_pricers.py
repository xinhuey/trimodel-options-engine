"""
Stage 4: automated tests 
prove that Monte Carlo and Binomial Tree converge to Black-Scholes, and that basic 
no-arbitrage identities (put-call parity) hold

Run with: pytest test_pricers.py -v

"""

import numpy as np 
import pytest

from black_scholes import black_scholes_price
from monte_carlo import monte_carlo_price
from binomial_tree import binomial_tree_price
from pricer import price 


# Just a handful of different option scenarios to test against
# Covers in-the-money, at-the-money and out-of-the-money

SCENARIOS = [
    # S,    K,  T,      r,  sigma
    (100,   100, 1.0, 0.05, 0.2),  # at-the-money, base case
    (110,   100, 1.0, 0.05,  0.2), # in-the-money call
    (90,   100, 1.0, 0.05,  0.2), # out-of-money call
    (100,   100, 0.25, 0.03,  0.35), # shorter maturity, higher vol
    (100,   100, 2.0, 0.05,  0.15) # longer maturity, lower vol 
]

@pytest.mark.parametrize("S,K,T,r,sigma", SCENARIOS)
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_monte_carlo_converges_to_black_scholes(S, K, T, r, sigma, option_type):
    """
    MC price should land within a tight tolerance of Black-Scholes
    at a large sample size.
    """
    bs_price = black_scholes_price(S, K, T, r, sigma, option_type)
    mc_result = monte_carlo_price(S, K, T, r, sigma, option_type, n_sims=500_000, seed=42)

    # Tolerance is a small multiple of the MC standard error
    # check that the estimate is consistent with the known answer 
    tolerance = 4 * mc_result["std_error"]
    assert abs(mc_result["price"] - bs_price) < tolerance,(
        f"MC price {mc_result['price']:.4f} too far from "
        f"BS price {bs_price:.4f} (tolerance {tolerance:.4f})"
    ) 

@pytest.mark.parametrize("S,K,T,r,sigma", SCENARIOS)
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_binomial_tree_converges_to_black_scholes(S, K, T, r, sigma, option_type):
    """
    European binomial tree price should converge tightly to 
    Black-Scholes as steps grows -- this method has no randomness, so we
    can use a much tighter tolerance than the MC test
    """
    bs_price = black_scholes_price(S, K, T, r, sigma, option_type)
    tree_price = binomial_tree_price(S, K, T, r, sigma, option_type, steps = 1000, american = False)

    assert abs(tree_price - bs_price) < 0.01,(
        f"Binomial tree price {tree_price:.4f} too far from "
        f"BS price {bs_price:.4f}"
    )

@pytest.mark.parametrize("S, K, T, r, sigma", SCENARIOS)
def test_put_call_parity(S, K, T, r, sigma):
    """
    C - P = S - K * exp(-rT) must hold for any correctly implemented
    Black-Scholes pricer - this is model free, no arbitrage identity,
    so a failure here means a bug, not just imprecision

    """
    call = black_scholes_price(S, K, T, r, sigma, "call")
    put = black_scholes_price(S, K, T, r, sigma, "put")

    lhs = call - put
    rhs = S - K * np.exp(-r * T)

    assert abs(lhs - rhs) < 1e-8, "Put-call parity violated"


def test_unified_interface_agrees_with_direct_calls():
    """
    pricer.py facade should return the exact same numbers as
    calling each underlying pricer directly 
    """
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2

    assert price("bs", S, K, T, r, sigma, "call") == black_scholes_price(S, K, T, r, sigma, "call")
    assert price("bt", S, K, T, r, sigma, "call", steps = 200) == binomial_tree_price(S, K, T, r, sigma, "call", steps = 200)

def test_invalid_method_raises():
    """
    The facade should fail loudly on a typo'd method name, not silently
    do the wrong thing
    """
    with pytest.raises(ValueError):
        price("not_a_real_method", 100, 100, 1.0, 0.05, 0.2, "call")
        