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

"""