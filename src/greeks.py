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