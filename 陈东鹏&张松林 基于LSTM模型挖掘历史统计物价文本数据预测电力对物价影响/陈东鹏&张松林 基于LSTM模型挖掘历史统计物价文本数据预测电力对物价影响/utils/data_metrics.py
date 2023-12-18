import numpy as np


def MAPE(y_true, y_pred):
    """# count The Mean Absolute Percent Error

    Disc：
        not * 100 ...
    """
    return np.mean(np.abs((y_pred - y_true) / y_true), axis=0)


def Dstat(y_true, y_pred):
    """# count Directional Statistic

    Disc：
        not * 100 ...
    """
    count = 0.0  # float
    for i in range(1, len(y_true)):
        if((y_true[i] - y_true[i-1]) * (y_pred[i] - y_true[i-1]) >= 0):  # not compare float
            count += 1

    return count / (len(y_true) - 1)