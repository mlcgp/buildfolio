def get_daterange(date_file):
    """
    Generate date range from file open
    """
    dates = open(date_file, 'r')
    ints = [int(val) for val in dates.read().split()]

    start = str(datetime.date((ints[0]), (ints[1]), (ints[2])))
    end = str(datetime.date((ints[3]), (ints[4]), (ints[5])))
    daterange = [start, end]
    return daterange


def alpha(self):
    """
    Get the risk free rate (US 1-year) and generate an excess return matrix

    :param returns: (daily) returns of stocks.
    :type returns: numpy array or pd.DataFrame
    :return: (daily) excess returns of stocks.
    :rtype: pd.DataFrame
    """
    r = self.returns()
    start = r.index.min()
    end = r.index.max()
    rf = np.divide(pdr.get_data_fred('DGS1', start=start,
                                     end=end).dropna(how="all"), 100)

    # Reshape the arrays if difference exists
    len_r = len(r)
    len_rf = len(rf)

    if len_r < len_rf:
        r = r.reindex(rf.index)
    elif len_r > len_rf:
        rf = rf.reindex(r.index)
    elif len_r == len_rf:
        pass

    r = r.astype(float)
    rf = rf.astype(float)

    # Construct the excess return array
    rshape = np.shape(r)
    rf = np.tile(rf, (1, rshape[1]))
    xs_returns = np.subtract(r, rf)

    return xs_returns
