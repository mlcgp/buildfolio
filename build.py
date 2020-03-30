"""
Main utility methods for portfolio automation.
"""

import pandas_datareader as pdr
import yfinance as yf
import numpy as np
import sys
from urllib.request import urlopen
from lxml import etree
from tqdm import tqdm
import pandas as pd
import os


class Portfolio:

    def __init__(self, ticker_file, start, end):
        """
        :param ticker_file: .txt file with a list of tickers.
        :type ticker_file: str
        :param start, end: user-defined dates.
        :type start, end: str
        """

        self.ticker_file = ticker_file
        self.tickers = self.tickers()
        self.n = len(self.tickers)
        self.start = start
        self.end = end
        self.w = self.init_weights()  # Starting portfolio weights in an array
        self.weights = self.weight_dict()  # Starting portfolio weights in a dictionary
        self.sectors = self.sectors()
        self.prices = self.prices()
        self.riskfree = self.riskfree()

    def tickers(self):
        """
        Create list for tickers on init.

        :return: ticker list.
        :rtype: list
        """
        # Read in the ticker list
        file_open = open(self.ticker_file).read()
        tickers = list(filter(None, list(map(str, file_open.split("\n")))))
        return tickers

    def init_weights(self):
        """
        Initial starting weights for the portfolio.

        :return: N array of ticker weights.
        :rtype: np.array
        """
        return np.ones((self.n), dtype='float64') / self.n

    def weight_dict(self, weights=None):
        """
        Ticker weight dictionary.

        :return: Ticker: weight dictionary
        :rtype: dictionary
        """
        if weights is None:
            w = list(self.w)
        if weights is not None:
            w = list(weights)
        self.weights = dict(zip(self.tickers, w))
        return self.weights

    def sectors(self):
        """
        Get the sector id for each ticker.

        :return: ticker dictionary.
        :rtype: dictionary
        """
        tickers = self.tickers

        urls = [
            f"https://ca.finance.yahoo.com/quote/{ticker}/profile?p={ticker}"
            for ticker in tickers]

        sector_id = []

        for url in tqdm(urls, desc='parsing', total=len(urls)):
            r = urlopen(url)
            htmlparser = etree.HTMLParser()
            tree = etree.parse(r, htmlparser)
            sector = tree.xpath(
                "normalize-space(//span[@class='Fw(600)']/text()[last()])")
            sector_id.append(sector)

        ticker_dict = dict(zip(tickers, sector_id))

        return ticker_dict

    def prices(self):
        """
        Get ticker prices from a list in a text file.

        :return: (daily) adjusted closing prices
        :rtype: pd.DataFrame
        """

        tickers = self.tickers
        start = self.start
        end = self.end

        for ticker in tqdm(tickers, desc='downloading', total=len(tickers)):
            prices = yf.download(tickers, start=start, end=end)['Adj Close']
            nan_sum = prices[ticker].isnull().sum(axis=0)
            if nan_sum > 1:
                sys.exit(f"""{nan_sum} days of data for {ticker} are missing.\n
                Try to reconsider the time frame.""")
            else:
                continue
        return prices

    def daily_returns(self):
        """
        Create return matrix from price dataframe.

        :return: (daily) returns of stocks.
        :rtype: pd.DataFrame
        """
        prices = self.prices
        daily_returns = prices.pct_change().dropna(how="all")
        return daily_returns

    def log_returns(self):
        """
        Create log return matrix from price dataframe.

        :return: (daily) log returns of stocks.
        :rtype: pd.DataFrame
        """
        prices = self.prices
        log_returns = np.log(prices).diff().dropna(how="all")
        return log_returns

    def annualized_returns(self, freq=252):
        """
        Generate an annualized return matrix

        :param freq: number of trading days in a year.
        :type freq: int
        :return: annualized mean daily returns.
        :rtype: pd.Series
        """
        r = self.daily_returns()
        return r.mean() * freq

    def riskfree(self):
        """
        Calculate the average risk-free rate over the portfolio's time horizon.

        :return: The average 1-year risk free rate over the portfolio's time horizon.
        :rtype: np.float64
        """
        r = self.daily_returns()
        start = r.index.min()
        end = r.index.max()
        rf = np.divide((pdr.get_data_fred('DGS1', start=start,
                                          end=end).dropna(how="all")).mean(), 100)
        return rf.iloc[0]

    def excess_returns(self):
        """
        Calculate the annualized return less the risk free rate.

        :return: annualized excess returns
        :rtype: pd.Series
        """
        r = self.annualized_returns()
        rf = self.riskfree

        # construct the risk free array and excess returns.
        rshape = np.shape(r)
        rf = np.tile(rf, rshape)
        excess = np.subtract(r, rf)
        return excess

    def ema_returns(self, freq=252, span=500):
        """
        The exponentially weighted moving average of the daily stock returns.
        A less noisy means for a price indicator.
        Using the ema returns gives more weight to recent data (parameterized
        by the span.)

        :param freq: number of trading days in a year.
        :type freq: int
        :param span: specify the decay for the span of the time period. (i.e.
        span=20 implies a 20-day ema) A higher value will imply a longer
        holding period and less rebalancing. A smaller value will imply a
        shorter time horizon for holdings.
        :type span: int
        :return: annualized mean ema returns.
        :rtype: pd.Series
        """
        r = self.daily_returns()
        ema_returns = r.ewm(span=span).mean().iloc[-1] * freq
        return ema_returns

    def excess_ema_returns(self):
        """
        Calculate the annualized ema return less the risk free rate.

        :return: annualized excess returns
        :rtype: pd.Series
        """
        r = self.ema_returns()
        rf = self.riskfree

        # construct the risk free array and excess returns.
        rshape = np.shape(r)
        rf = np.tile(rf, rshape)
        excess = np.subtract(r, rf)
        return excess

    @staticmethod
    def save_to_csv(array, name):
        """
        Save an array to csv format.

        :param array: A price or return dataframe.
        :type array: pd.DataFrame, np.array
        :param name: Desired filename.
        :type name: str
        :return: csv file in the current working directory.
        :rtype: .csv
        """
        cwd = os.getcwd()
        filename = f"{cwd}/{name}.csv"

        if isinstance(array, np.ndarray or np.array):
            new_file = np.savetxt(filename, array, delimiter=",")
            return new_file

        if isinstance(array, pd.DataFrame):
            pandas = pd.DataFrame(array)
            new_file = pandas.to_csv(filename)
            return new_file
