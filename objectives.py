import numpy as np
import pandas as pd


def mean(w, returns):
    """
    Calculate the expected return for the portfolio.

    :param w: Portfolio weights.
    :type w: np.array
    :param returns: Portfolio's expected returns.
    :type returns: pd.Series
    :return: Portfolio return.
    :rtype: np.float64
    """
    if isinstance(w, pd.DataFrame):
        w = w.to_numpy()
    if isinstance(returns, pd.DataFrame):
        returns = returns.to_numpy()
    return np.dot(w.T, returns)


def var(w, cov):
    """
    Calculate the variance of the portfolio.

    :param w: Portfolio weights.
    :type w: np.array
    :param cov: Portfolio covariance matrix.
    :type cov: np.ndarray
    :return: Variance of the portfolio.
    :rtype: np.float64
    """
    if isinstance(w, pd.DataFrame):
        w = w.to_numpy()
    if isinstance(cov, pd.DataFrame):
        cov = cov.to_numpy()
    return np.dot(np.dot(w.T, cov), w)


def sharpe(mu, var, rf_rate):
    """
    Calculate the Sharpe ratio for the portfolio.

    :param mu: Portfolio return
    :type mu: np.float64, float
    :param var: Portfolio variance
    :type var: np.float64, float
    :param rf_rate: Portfolio risk-free rate
    :type rf_rate: np.float64, float
    :return: Portfolio sharpe ratio
    :rtype: np.float64
    """
    return (mu - rf_rate) / np.sqrt(var)
