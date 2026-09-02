# TriModel Options Engine

A Python options pricing engine that prices European options three independent ways — **Black-Scholes** (closed-form), **Monte Carlo simulation**, and a **Binomial Tree**. It then cross-validates the results against each other and computes the Greeks.

## The decision to have three options:

Any one of these on its own is a copy-paste exercise. Building all three and proving they agree is what actually demonstrates understanding of option pricing theory:

- **Black-Scholes** is the closed-form "ground truth" for European options.
- **Monte Carlo** simulates thousands of random price paths under the risk-neutral measure and averages the discounted payoffs. It's slower, but generalizes to path-dependent and exotic payoffs that Black-Scholes can't handle.
- **Binomial Tree** builds the price out step-by-step via backward induction, and naturally supports American-style early exercise.

As the number of Monte Carlo simulations or binomial tree steps grows, both converge to the Black-Scholes price. This project shows that convergence explicitly rather than just asserting it.

## Features

- Price European call & put options via Black-Scholes, Monte Carlo, and Binomial Tree
- Analytical Greeks (Delta, Gamma, Vega, Theta, Rho) from Black-Scholes
- Finite-difference Greeks as an independent cross-check
- Monte Carlo confidence intervals + variance reduction (antithetic variates)
- Convergence plots (MC and Binomial → Black-Scholes as N grows)
- Option price surface and payoff diagram visualizations
- Built-in sanity checks (e.g. put-call parity) at every stage

<!-- ## Project structure

```
trimodel-options-engine/
├── src/
│   ├── black_scholes.py      # closed-form pricer + analytical Greeks
│   ├── monte_carlo.py        # MC pricer + variance reduction
│   ├── binomial_tree.py      # CRR binomial tree
│   ├── greeks.py             # finite-difference Greeks, cross-checked vs. analytical
│   ├── plots.py              # payoff diagrams, price surface, convergence plots
│   └── pricer.py             # unified interface across all three methods
├── notebooks/
│   └── demo.ipynb            # walkthrough notebook
├── tests/
│   └── test_pricers.py       # MC/Binomial converge to Black-Scholes within tolerance
├── app.py                    # Streamlit dashboard
├── requirements.txt
└── README.md
``` -->

## Installation

```bash
git clone https://github.com/<your-username>/options-pricing-engine.git
cd options-pricing-engine
pip install -r requirements.txt
```

**Requirements:** Python 3.9+, `numpy`, `scipy`, `matplotlib` (and `streamlit` if using the dashboard).

## Usage

```python
from src.black_scholes import black_scholes_price, black_scholes_greeks

S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2

price = black_scholes_price(S, K, T, r, sigma, option_type="call")
greeks = black_scholes_greeks(S, K, T, r, sigma, option_type="call")

print(f"Call price: {price:.4f}")
print(greeks)
```

Once the other pricers are built, the unified interface will look like:

```python
from src.pricer import price

price(method="bs", S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="call")
price(method="mc", S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="call", n_sims=100_000)
price(method="binomial", S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="call", steps=200)
```

## Example output

| Method                    | Call Price  | Put Price  |
| ------------------------- | ----------- | ---------- |
| Black-Scholes             | 10.4506     | 5.5735     |
| Monte Carlo (100k sims)   | ~10.45 ± CI | ~5.57 ± CI |
| Binomial Tree (200 steps) | ~10.45      | ~5.57      |

**Greeks (Black-Scholes, call):** Delta 0.6368 · Gamma 0.0188 · Vega 37.524 · Theta -6.414 · Rho 53.233

## Formulae

**Black-Scholes:**

```
d1 = (ln(S/K) + (r + 0.5*sigma^2)*T) / (sigma*sqrt(T))
d2 = d1 - sigma*sqrt(T)
Call = S*N(d1) - K*exp(-r*T)*N(d2)
Put  = K*exp(-r*T)*N(-d2) - S*N(-d1)
```

**Monte Carlo (GBM under the risk-neutral measure):**

```
S_T = S * exp((r - 0.5*sigma^2)*T + sigma*sqrt(T)*Z),   Z ~ N(0,1)
Price = exp(-r*T) * mean(payoff(S_T))
```

**Binomial Tree (Cox-Ross-Rubinstein):**

```
u = exp(sigma*sqrt(dt)),  d = 1/u
p = (exp(r*dt) - d) / (u - d)
Backward induction: V = exp(-r*dt) * [p*V_up + (1-p)*V_down]
```

<!-- ## Roadmap

- [x] Stage 1 — Black-Scholes pricer + analytical Greeks
- [ ] Stage 2 — Monte Carlo pricer with confidence intervals + variance reduction
- [ ] Stage 3 — Binomial Tree (European + American)
- [ ] Stage 4 — Unified pricer interface + cross-validation tests
- [ ] Stage 5 — Finite-difference Greeks cross-check
- [ ] Stage 6 — Visualizations (price surface, payoff diagrams, convergence plots)
- [ ] Stage 7 — Streamlit dashboard -->

<!-- **Stretch goals:** Asian/barrier options, implied volatility solver, Heston stochastic volatility model, live market data comparison via `yfinance`. -->

<!-- ## Skills demonstrated

Python · NumPy/SciPy · Probability & stochastic processes · Numerical methods (finite differences, tree methods, Monte Carlo convergence) · Financial mathematics (options, Greeks, no-arbitrage pricing) · Software design (clean interfaces, unit testing) -->

## License

MIT
