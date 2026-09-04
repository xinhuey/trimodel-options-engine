"""
Stage 3: Binomial Tree option pricer (CRR model)

Prices option by discretizing time into 'step' intervals
lets the stock move up or down by a fixed factor at each step 
then work backwards from the known payoffs at expiry to find today's price

Unlike Black-Scholes and plain Monte Carlo, this method naturally supports
American-style early exercise
at each backward step we can check whether exercising immediately beats holding 

Expectation: should converge to black_scholes.py's output at 'steps' grows for the 
European options 

Note: American is not used here yet, it is only used for European options

"""

import numpy as np 

from black_scholes import black_scholes_price

def binomial_tree_price(S, K, T, r, sigma, option_type="call", steps = 200, american = False):
    """
    Price an option via the CRR binomial tree 

    Args:
        american: if True, allow early exercise at every node 
                  if False, only exercise at expiry 
    
    Returns:
        float: option price

    """
    dt = T / steps 

    # Up / down factors -- CRR's specific choice ensures the tree recombines
    # (an up-then-down move lands on the same price as down-then-up)
    # This keeps the tree's size manageable: N + 1 nodes at step N, not 2^N

    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u

    # Risk-neutral probability of an up-move. Same risk-neutral idea as
    # Black-Scholes and Monte Carlo, discretized into one step 

    p = (np.exp(r * dt) - d) / (u - d)

    discount = np.exp(-r * dt)

    # Step 1 : stock prices at expiry (tree's final layer)
    # After 'steps' steps, there are 'steps + 1' possible outcomes 
    # indexed by how many "up" moves occurred(j = 0 .. steps)

    j = np.arange(steps + 1)
    S_T = S * (u ** j) * (d ** (steps - j))

    # Step 2: payoffs at expiry 
    if option_type == "call":
        values = np.maximum(S_T - K, 0)
    elif option_type == "put":
        values = np.maximum(K - S_T, 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    
    # Step 3: work backwards through the tree 
    # At each earlier step, every node's value is the discounted
    # probability-weighted average of its two children (up-child, down-child)
    for step in range(steps -1, -1, -1):
        values = discount * (p * values[1:] + (1 -p) * values[:-1])

    return values[0]


if __name__ == "__main__":
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2

    bs_call = black_scholes_price(S, K, T, r, sigma, "call")
    bs_put = black_scholes_price(S, K, T, r, sigma, "put")

    print(f"Black-Scholes call price(ground truth): {bs_call:.4f}")
    print(f"Black Scholes put price (ground truth): {bs_put:.4f}")

    print("Convergence Check -- European options should approach")
    print("Black-Scholes as the number of steps grow")

    for steps in [10, 50, 200, 1000]:
        euro_call = binomial_tree_price(S, K, T, r, sigma, "call", steps = steps, american = False)
        euro_put = binomial_tree_price(S, K, T, r, sigma, "put", steps = steps, american = False)

        print(
            f"steps={steps:>5}  "
            f"call: {euro_call:.4f} (error {abs(euro_call - bs_call):.4f})  "
            f"put: {euro_put:.4f} (error {abs(euro_put - bs_put):.4f})"
        )