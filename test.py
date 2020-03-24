from build import Builder
import objectives as obj
import matrix
import optimizers as opt
import numpy as np

file = "tickers.txt"

portfolio = Builder(file, "2018-01-01", "2019-01-01")  # Portfolio constructor
p = portfolio.prices
daily = portfolio.daily_returns()
log = portfolio.log_returns()
r = portfolio.annualized_returns()
c = matrix.LedoitWolf(daily)
rf = portfolio.riskfree
w = portfolio.w
mu = obj.mean(w, r)
var = obj.var(w, c)
sharpe = obj.sharpe(mu, var, rf)

# Find the minimum variance portfolio
print(opt.min_var(p, w, c))
print(opt.min_mad(log))
print(opt.max_sharpe(p, w, r, c, rf))
