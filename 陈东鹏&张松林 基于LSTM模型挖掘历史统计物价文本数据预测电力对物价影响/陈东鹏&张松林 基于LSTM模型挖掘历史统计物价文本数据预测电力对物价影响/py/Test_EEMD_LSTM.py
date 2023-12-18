import numpy as np
import pandas
import pylab as plt  # matplotlib的一个子包
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
from sklearn.utils import shuffle
from scipy.sparse import coo_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error # 平方绝对误差
from sklearn.metrics import r2_score  # R square

import sys 
sys.path.append(r'F:\workplace\Python\ml\LSTM-Agricultural-Products-Prices\Time-Series-Prediction-with-LSTM/')  # 要用绝对路径
from utils import eemd_tools, data_tools, networks_factory, data_metrics
from utils.constants import const


# fix random seed for reproducibility
np.random.seed(7)


data_multi = np.load(const.PROJECT_DIR + "data/eemd/apple/data_multi.npy")
print("# shape", data_multi.shape)  # not .shape()
# print(data_multi)
n_dims = data_multi.shape[1]  # magic number !
print("# dims: ", n_dims)


# normalize features
scaler = data_tools.Po_MinMaxScaler
scaled = scaler.fit_transform(data_multi)

output = 1
lag = const.LOOK_BACK

reframed = data_tools.series_to_supervised(scaled, lag, output)
# drop columns we don't want to predict
index_drop = [-j-1 for j in range(data_multi.shape[1] - 1)]
reframed.drop(reframed.columns[index_drop], axis=1, inplace=True)
data_supervised = reframed.values
print("# shape:", reframed.shape)
print(len(data_multi) == len(reframed) + lag)
# print(reframed.head(3))

# split into train and test sets
train_size = int(len(data_supervised) * const.TRAIN_SCALE)
test_size = len(data_supervised) - train_size
train_data, test_data = data_supervised[0:train_size,:], data_supervised[train_size:len(data_multi),:]
print(len(train_data), len(test_data))
print(len(data_supervised) == len(train_data) + len(test_data)) 
# print(train_data)


# split into input and outputs
train_X, train_Y = train_data[:, :-1], train_data[:, -1]
test_X, test_Y = test_data[:, :-1], test_data[:, -1]
print("# shape:", train_X.shape)
print("# shape:", train_Y.shape)


from sklearn.utils import shuffle
from scipy.sparse import coo_matrix

# shuffle train set (include validation set)
trainX_sparse = coo_matrix(train_X)  # sparse matrix
train_X, trainX_sparse, train_Y = shuffle(train_X, trainX_sparse, train_Y, random_state=0)


time_steps = lag
n_lstm_neurons = [8, 16, 32, 64, 128]
# n_lstm_neurons = [8]  # for once
n_epoch = networks_factory.EPOCHS
n_batch_size = networks_factory.BATCH_SIZE


# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], time_steps, train_X.shape[1]//time_steps))
test_X = test_X.reshape((test_X.shape[0], time_steps, test_X.shape[1]//time_steps))
print(train_X.shape, train_Y.shape)
print(test_X.shape, test_Y.shape)


for i, n_lstm_neuron in enumerate(n_lstm_neurons):
    
    print("-----------n_lstm_neuron: %d--------------" % n_lstm_neuron)
    
    s, model = networks_factory.create_lstm_model_dropout(lstm_neurons=n_lstm_neuron, hidden_layers=2, 
                                                          lenth=time_steps, dims=n_dims, n_out=1)
    model.compile(loss='mean_squared_error', optimizer='adam')
    history = model.fit(train_X, train_Y, epochs=10, batch_size=n_batch_size, validation_split=const.VALIDATION_SCALE,
                    verbose=0, callbacks=[networks_factory.ES])  # callbacks=[networks_factory.ES]
    print("# Finished Training...")
    
    # make a prediction
    train_predict = model.predict(train_X)
    test_predict = model.predict(test_X)
                                                    
    # invert predictions
    inv_trainP, inv_trainY = data_tools.inv_transform_multi(scaler, train_X, train_predict, train_Y)
    inv_testP, inv_testY = data_tools.inv_transform_multi(scaler, test_X, test_predict, test_Y)

    # calculate RMSE, MAPE, Dstat
    train_rmse = sqrt(mean_squared_error(inv_trainP, inv_trainY))
    test_rmse = sqrt(mean_squared_error(inv_testP, inv_testY))
    print('Train RMSE: %.4f, Test RMSE: %.4f' % (train_rmse, test_rmse))
    train_mape = data_metrics.MAPE(inv_trainP, inv_trainY)
    test_mape = data_metrics.MAPE(inv_testP, inv_testY)
    print('Train MAPE: %.4f, Test MAPE: %.4f' % (train_mape, test_mape))
    train_ds = data_metrics.Dstat(inv_trainP, inv_trainY)
    test_ds = data_metrics.Dstat(inv_testP, inv_testY)
    print('Train Dstat: %.4f, Test Dstat: %.4f' % (train_ds, test_ds))
    
print("# All Done!")