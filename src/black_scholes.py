"""
Stage 1: Black-Scholes closed-form option pricer + analytical Greeks

This module is the "ground truth" for the rest of the project
- Monte Carlo and Binomial Tree pricers will be validated against this 

Notation:
    S       : current spot price of the underlying
    K       : strike price
    T       : time to maturity, in years
    r       : risk-free interest rate (annualized, continuously compounded)
    sigma   : volatility of the underlying (annualized)
    option_type : "call" or "put"



"""

import numpy as np 
from scipy.stats import norm 

def _d1_d2(S, K, T, r, sigma):
    """
    Compute d1 and d2, the two intermediate terms in the BS formula
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2

def black_scholes_price(S, K, T, r, sigma, option_type="call"):
    """
    Price a European option using the Black-scholes closed-form formula

    Returns:
        float: option_price
    
    
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma)

    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return price 

def black_scholes_greeks(S, K, T, r, sigma, option_type="call"):
    """
    Compute the main Greeks analytically: Delta, Gamma, Vega, Theta, Rho

    Note on units:
        - Vega is returned per 1.00(100%) change in sigma; divide by 100
        for "per 1% vol point" convention if you want that instead.

        - Theta is returned per YEAR; divide by 365 for "per day" decay.

        - Rho is returned per 1.00 (100%) change in r; divide by 100 for 
        "per 1% rate point" convention
    
    """
    d1, d2 = _d1_d2(S, K, T,r, sigma)
    pdf_d1 = norm.pdf(d1)

    # Delta and Rho differ between calls and puts; Gamma and Vega do not 
    if option_type == "call":
        delta = norm.cdf(d1)
        theta = (
            -(S * pdf_d1 * sigma) / (2 * np.sqrt(T))
            - r * K * np.exp(-r * T) * norm.cdf(d2)
        )

        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    
    elif option_type == "put":
        delta = norm.cdf(d1) - 1
        theta = (
            -(S * pdf_d1 * sigma) / (2 * np.sqrt(T))
            + r * K * np.exp(-r * T) * norm.cdf(-d2)
        )
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    gamma = pdf_d1 / (S * sigma * np.sqrt(T))
    vega = S * pdf_d1 * np.sqrt(T)

    return{
        "delta": delta,
        "gamma": gamma,
        "vega": vega, 
        "theta": theta,
        "rho": rho,
    }

if __name__ == "__main__":
    # Sanity check - try an at-the-money-call
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2

    call_price = black_scholes_price(S, K, T, r, sigma, "call")
    put_price = black_scholes_price(S, K, T, r, sigma, "put")
    greeks = black_scholes_greeks(S, K, T, r, sigma, "call")

    print(f"Call price: {call_price:.4f}")
    print(f"Put price: {put_price:.4f}")
    print("Call Greeks:")
    for name, value in greeks.items():
        print(f" {name.capitalize():6s}: {value:.4f}")

    # Check: put-call parity should hold
    # C - P = S - K * exp(-r * T)
    lhs = call_price - put_price
    rhs = S - K * np.exp(-r * T)
    print(f"\nPut-call parity check: {lhs:.4f} vs {rhs:.4f} (should match)")

