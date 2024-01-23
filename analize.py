from statsmodels.tsa.stattools import adfuller


def check_stat(data):
    result = adfuller(data)
    return result[1] < 0.05


