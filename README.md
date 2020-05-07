![logo-banner](/Users/malcolmgillespie/Projects/buildfolio/media/logo-banner.png)A starting point for your investment portfolio.

<p align="left">
    <a href="https://www.python.org/">
        <img src="https://ForTheBadge.com/images/badges/made-with-python.svg"
            alt="python"></a>
</p>

Example

``` python
from build import Portfolio
from matrix import LedoitWolf
from optimizers import min_var
from objectives import sharpe, var, mean

# import a file with a list of tickers
file = "sample_tickers.txt"

# Constructer to access a variety of methods
portfolio = Portfolio(file, "2018-01-01", "2019-01-01")
px = portfolio.prices

# If you wanted to save a Numpy or Pandas array to a csv file, that is possible
Portfolio.save_to_csv(px, "prices")

# Get portfolio variables
daily = portfolio.daily_returns()
r = portfolio.annualized_returns()
c = LedoitWolf(daily)
rf = portfolio.riskfree
w = portfolio.w # initial portfolio weights
var = var(w, c)
mu = mean(w, r)

# Optimize for the minimum variance portfolio
new_weights = min_var(px, w, c)

# Append the new weights to the portfolio instance
print(portfolio.weight_dict(min_var_weights))

# Can now call the portfolio weights
print(portfolio.weights)
```

```python
{'MSFT': 0.34207696134482046, 'SBUX': 0.16711385165240025, 'INTL': 0.48900343581850647, 'TSLA': 0.001805751184272882}
```

```python
# Offers a method for sector classifications using a simple web scraping script
print(portfolio.sectors)
```

```python
{'MSFT': 'Technology', 'SBUX': 'Consumer Cyclical', 'INTL': 'Financial Services', 'TSLA': 'Consumer Cyclical'}
```



## Modules

Documentation for buildfolio methods can be found in py docstrings.

### build

```python
Portfolio.tickers
Portfolio.sectors
Portfolio.prices
Portfolio.daily_returns
Portfolio.log_returns
Portfolio.annualized_returns
Portfolio.riskfree
Portfolio.excess_returns
Portfolio.ema_returns
Portfolio.excess_ema_returns
Portfolio.save_to_csv
```

### matrix

```python
matrix.sample_cov
matrix.ema_cov
matrix.LedoitWolf
matrix.format_as_df
```

### objectives

```python
objectives.mean
objectives.var
objectives.sharpe
```

### optimizers

```python
optimizers.min_var
optimizers.min_mad
optimizers.max_sharpe
```

## Backlog

- Capital allocation module
- Portfolio backtest/visualization module
- Portfolio performance module
- Hierarchical Risk Parity optimization process
- Monte Carlo optimization process
- CVaR optimization process
- Omega Ratio
- Sortino Ratio



**Disclosure**: 

*This is not professional trading or investment advice. Investors are reminded that they should seek advice from a registered broker or financial advisor before making any investment decisions. This material is considered general information, and is not to be relied on as a formal investment recommendation.*

