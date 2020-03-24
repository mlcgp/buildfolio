import numpy as np
import pandas as pd


def sample_cov(daily_returns, freq=252):
    """
    Basic annualized sample covariance matrix.

    :param returns: (daily) returns for the portfolio.
    :type returns: pd.DataFrame
    :param freq: number of trading days in a year.
    :type freq: int
    :return: covariance matrix
    :rtype: np.ndarray
    """
    return daily_returns.cov().to_numpy() * freq


def ema_cov(daily_returns, span=180, freq=252):
    """
    Exponentially weighted covariance matrix. (annualized)

    :param returns: (daily) returns for the portfolio.
    :type returns: pd.DataFrame
    :param freq: number of trading days in a year.
    :type freq: int
    :return: EWMA annualized covariance matrix
    :rtype: np.ndarray
    """
    cov = daily_returns.ewm(span=span).cov()
    cov = cov.iloc[-len(daily_returns.columns):]
    cov = cov.reset_index(level=0, drop=True)
    return cov.to_numpy() * freq


def LedoitWolf(daily_returns, annualized=True, freq=252):
    """
    Shrinkage estimator algorithm proposed by Ledoit and Wolf.
    Implemented as specified in the paper.

    :param daily_returns: (daily) returns for the portfolio.
    :type daily_returns: pd.DataFrame
    :param annualized: Annualize the returned matrix. Defaults to True.
    :type annualized: boolean
    :param freq: number of trading days in a year.
    :type freq: int
    :return: Shrunk covariance matrix.
    :rtype: np.ndarray
    """
    R = daily_returns.to_numpy()
    t, n = np.shape(R)
    S = daily_returns.cov()

    # Constant correlation target
    var = np.diag(S).reshape(-1, 1)
    std = np.sqrt(var)
    _std = np.tile(std, (n,))
    s_ii = np.tile(var, (n,))
    s_jj = s_ii.T
    r_ij = S / np.sqrt(s_ii * s_jj)
    r_bar = (np.sum(np.sum(r_ij)) - n) / (n*(n-1))
    F = r_bar * np.sqrt(s_ii * s_jj)

    # Estimate pi
    Xm = R - R.mean(axis=0)
    y = Xm ** 2
    pi_hat = np.dot((1.0 / t), np.sum(np.sum(np.dot(y.T, y))))-np.sum(np.sum(S**2.0))

    # Estimate rho (separated into terms) (borrowed heavily from online sources)
    rdiag = np.dot((1.0 / t), sum(sum(y**2.0))) - sum(var**2.0)
    v = np.dot((Xm**3.0).T, Xm) / t - ((var*S).T)
    v = v.to_numpy()
    np.fill_diagonal(v, 0.0)
    roff = sum(sum(v * (_std / _std.T)))
    rho_hat = rdiag + np.dot(r_bar, roff)

    # Estimate gamma
    gamma_hat = np.linalg.norm(S - F, 'fro') ** 2

    # Estimate k
    k_hat = (pi_hat - rho_hat) / gamma_hat

    # Estimate lambda hat
    delta_hat = max(0.0, (min(1.0, (k_hat / t))))

    # Shrink the matrix
    S = S.to_numpy()
    shrunk_cov = (delta_hat * F) + ((1 - delta_hat) * S)

    # Annualized the covariance matrix
    if annualized is True:
        shrunk_cov = shrunk_cov * freq
    elif annualized is False:
        shrunk_cov = shrunk_cov

    return shrunk_cov


def format_as_df(prices, matrix):
    """
    Format as pandas DataFrame.

    :param prices: Portfolio prices dataframe.
    :type prices: pd.DataFrame
    :param matrix: Portfolio covariance matrix.
    :type matrix: np.array, np.ndarray
    :return: Covariance matrix in a DataFrame.
    :rtype: pd.DataFrame
    """
    # Format the numpy covariance matrix
    columns = prices.columns
    matrix = pd.DataFrame(matrix, columns=columns, dtype='float64')
    matrix = matrix.set_index(columns)
    return matrix
