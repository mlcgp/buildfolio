import numpy as np
import objectives as obj
import scipy.optimize as sco


def min_var(prices, w, cov, lb=0.0, ub=1.0):
    """
    Minimum-variance portfolio.

    :param prices: (daily) prices of stocks.
    :type prices: pd.DataFrame
    :param w: Portfolio weights.
    :type w: np.array
    :param cov: Portfolio covariance matrix.
    :type cov: np.ndarray
    :param lb, ub: Set the lower bound and upper bound for the portfolio
    allocations. Defaults are set to (0.0, 1.0).
    :type lb, ub: float
    :return: Minimum variance portfolio.
    :rtype: np.array
    """
    bounds = (lb, ub)
    n = len(w)

    """
    Normalize asset returns

    Set up a constraint so that the sum of the normalized returns
    weighted by the portfolio allocations needs to be equal to or greater
    than a threshold. (To ensure more weight to assets with positive
    risk adjusted returns.)
    """
    pmean = prices.pct_change().mean()
    pstd = prices.pct_change().std()
    pnorm = pmean / pstd
    pnorm = pnorm.values

    def objective(w=w, cov=cov):
        return obj.var(w, cov)

    init_guess = np.ones(n, dtype='float64') / n
    method = 'SLSQP'
    threshold = 0.02
    bnds = [bounds for i in range(n)]
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(np.abs(w))-1},
            {'type': 'ineq', 'fun': lambda x:  np.dot(x, pnorm)-threshold})
    result = sco.minimize(
        objective,
        x0=init_guess,
        method=method,
        bounds=bnds,
        constraints=cons
    )
    result = result["x"]
    return result


def min_mad(log_returns, lb=0.0, ub=1.0):
    """
    Minimum Mean absolute deviation portfolio.

    This optimization method requires no covariance matrix, does not assume the
    normality of returns and is more computationally efficient than the quadratic
    optimization routine of the Markowitz style portfolio.

    :param log_returns: Portfolio's daily log returns.
    :type log_returns: pd.DataFrame
    :return: Minimum MAD portfolio.
    :rtype: np.array
    """
    bounds = (lb, ub)
    n = len(log_returns.columns)

    def objective(w, log_returns):
        return (log_returns - log_returns.mean()).dot(w).abs().mean()
    init_guess = np.ones(n) / n
    bnds = [bounds for i in range(n)]
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(np.abs(x))-1})
    result = sco.minimize(
        objective,
        x0=init_guess,
        args=log_returns,
        bounds=bnds,
        constraints=cons
    )
    result = result["x"]
    return result


def max_sharpe(prices, w, returns, cov, rf_rate, lb=0.0, ub=1.0):
    """
    Solve for the maximum Sharpe (tangency) portfolio.

    :param prices: (daily) prices of stocks.
    :type prices: pd.DataFrame
    :param w: Portfolio weights.
    :type w: np.array
    :param returns: Portfolio's expected returns.
    :type returns: pd.Series
    :param cov: Portfolio covariance matrix.
    :type cov: np.ndarray
    :param rf_rate: The riskfree rate for the portfolio
    :type rf_rate: float, np.float64
    :param lb, ub: Set the lower bound and upper bound for the portfolio
    allocations. Defaults are set to (0.0, 1.0).
    :type lb, ub: float
    :return: Maximum Sharpe portfolio.
    :rtype: np.array
    """
    bounds = (lb, ub)
    n = len(prices.columns)

    def objective(w=w, returns=returns, cov=cov, rf_rate=rf_rate):
        mu = obj.mean(w, returns)
        var = obj.var(w, cov)
        sharpe = obj.sharpe(mu, var, rf_rate)
        return -sharpe

    init_guess = np.ones(n, dtype='float64') / n
    method = 'SLSQP'
    bnds = [bounds for i in range(n)]
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(np.abs(w))-1})
    result = sco.minimize(
        objective,
        x0=init_guess,
        method=method,
        bounds=bnds,
        constraints=cons
    )
    result = result["x"]
    return result
