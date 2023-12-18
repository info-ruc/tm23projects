import numpy
import pandas
from pandas import DataFrame
from pandas import Series
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler


# commonly used class
Ne_MinMaxScaler = MinMaxScaler(feature_range=(-1, 1))
Po_MinMaxScaler = MinMaxScaler(feature_range=(0, 1))


def load_data_from_csv(filename):
    """# load_data_from_csv

    Disc：
        price, column...
    """
    # load pima indians dataset
    dataframe = pandas.read_csv(filename, engine='python')
    dataframe = dataframe[['date', 'price']]
    dataset = dataframe.values  # Return a Numpy representation of the DataFrame

    prices, dates = dataset[:,1], dataset[:,0]
    prices = prices.astype('float32')  # 变为 float type
    prices = prices.reshape((len(dataset), 1))  # two dims

    return prices, dates


# -------------------------------------------------------- #
# Univariate one-step                                      #
# -------------------------------------------------------- #


def create_dataset(dataset, look_back=1):
    """# convert an array of values into a dataset matrix

    Disc：
        univariate...
    """
    dataX, dataY = [], []
    # 原文-1错了，"for i in range(n)"为[0,n-1]
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back), 0]  # 不包括i+look_back
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])  # 单值
    return numpy.array(dataX), numpy.array(dataY)


def create_mutil_dataset(train, imf_fine, imf_coarse, resdiue, look_back=1):
    """# convert an array of values into a dataset matrix

    Disc：
        NOT RECOMMEND, use series_to_supervised(data, n_in=1, n_out=1, dropnan=True)...
    """
    dataX, dataX1, dataY = [], [], []
    for i in range(len(imf_fine)-look_back):  # 原文-1错了，"for i in range(n)"为[0,n-1]
        fi = imf_fine[i:(i+look_back)]  # 不包括i+look_back
        co = imf_coarse[i:(i+look_back)]  # 不包括i+look_back
        res = resdiue[i:(i+look_back)]  # 不包括i+look_back
        dataX1 = np.vstack((fi, co, res))
        dataX1 = dataX1.T
        if(i == 1):
            print(fi.shape)
            print(dataX1.shape)
        dataX.append(dataX1)
        dataY.append(train[i + look_back])  # 单值
    return np.array(dataX), np.array(dataY)


# -------------------------------------------------------- #
# Difference series                                        #
# -------------------------------------------------------- #


def difference(dataset, interval=1):
    """# create a differenced series

    Disc：
        univariate...
    """
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)


def inverse_difference(last_ob, forecast):
    """# invert differenced forecast

    Disc：
        univariate...
    """
       # invert first forecast
    inverted = list()
    inverted.append(forecast[0] + last_ob)
       # propagate difference forecast using inverted first value
    for i in range(1, len(forecast)):
        inverted.append(forecast[i] + inverted[i-1])
    return inverted


# -------------------------------------------------------- #
# Series to supervised (both univariate and multivariate   #
# -------------------------------------------------------- #


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """# convert time series into supervised learning problem

    Disc：
        both univariate and multivariate...
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pandas.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg  # or agg.values(numpy array)


def inv_transform_multi(scaler, test_X, yhat, test_y):
    """# inv_transform for multivariate

    Disc：
        yhat: predict value
        test_y: the ground truth
    """
    n_in, n_dims = test_X.shape[1], test_X.shape[2]
    # invert scaling for forecast
    test_X = test_X.reshape((test_X.shape[0], n_in*n_dims))
    inv_yhat = numpy.concatenate((yhat, test_X[:, 1:n_dims]), axis=1)  # axis=1, Add to the end of the column
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]
    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = numpy.concatenate((test_y, test_X[:, 1:n_dims]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,0]
    return inv_yhat, inv_y


# -------------------------------------------------------- #
# Forecast by month data                                   #
# -------------------------------------------------------- #


def extract_features_to_day(dataframe):
    """# extract statistic features; not recommend

    Disc：
        ...
    """
    first_date = dataframe.iloc[0][0]
    last_date = dataframe.iloc[-1][0]
    print("# date: ", first_date, ", ", last_date)

    dataframe['date'] = pandas.to_datetime(dataframe['date'])
    dataframe = dataframe.set_index('date')

    df_month_mean = dataframe.resample('MS').mean()  # not resample('MS')['price'], to concat
    # NOTE that resample('MS') only has the 1st day of the month
    df_last_date = dataframe.tail(1)
    df_last_date.iloc[0][0] = df_month_mean.iloc[-1][0]
    df_month_mean = df_month_mean.append(df_last_date, ignore_index=False)
    print("\n# df_month_mean:")
    print(df_month_mean.head(3))
    print(df_month_mean.tail(2))

    df_month_mean = df_month_mean.asfreq('D', method= 'ffill')
    print("\n# df_month_mean asfreq:")
    print(df_month_mean.head(3))
    print(df_month_mean.tail(2))

    df_month_mean = df_month_mean[first_date:last_date]
    df_month_mean = df_month_mean.reset_index('date')
    dataframe = dataframe.reset_index('date')
    dataframe['price'] = df_month_mean['price']
    print("\n# dataframe:")
    print(dataframe.head(3))
    print(dataframe.tail(2))

    return dataframe


def extract_features_to_month(dataframe):
    """# extract statistic features

    Disc：
        ...
    """
    dataframe['date'] = pandas.to_datetime(dataframe['date'])
    dataframe = dataframe.set_index('date')

    # statistic features: methods are from computation module
    df_month_mean = dataframe.resample('MS').agg(['mean', 'median', 'min', 'max', 'std', 'var'])
    df_month_mean = df_month_mean.to_period('M')
    # print(df_month_mean.head(3))
    # print(df_month_mean.tail(2))

    dataset = df_month_mean.values
    # print("\n# shape:", dataset.shape)

    # ont-hot code for months
    dates = df_month_mean.reset_index('date')['date']
    # PeriodIndex.month. Note: return '1' instead of '01', so use int()
    dates_month = dates.apply(lambda x: int(x.month))
    dates_month = dates_month.values.T
    # print(dates_month)

    lb = LabelBinarizer()
    months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    lb.fit(months)
    onehot_month = lb.transform(dates_month)
    # print("\n# shape:", onehot_month.shape)

    return dataset, onehot_month
