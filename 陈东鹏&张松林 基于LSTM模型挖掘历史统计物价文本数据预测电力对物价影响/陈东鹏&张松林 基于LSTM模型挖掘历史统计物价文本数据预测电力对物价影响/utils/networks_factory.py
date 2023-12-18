from keras.models import Sequential
from keras.engine.input_layer import Input
from keras.layers import Embedding, SimpleRNN
from keras.layers import Dense, LSTM, BatchNormalization
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras import optimizers

import tensorflow as tf


# networks parameters
EPOCHS = 2000
BATCH_SIZE = 32  # default to 32
HIDDEN_LAYERS = 2
HIDDEN_NEURONS = 32
ES = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)  # patience=EPOCHS*0.1
SGD = optimizers.SGD(lr=0.01, clipvalue=0.5)
RMSprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
Adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)


def create_lstm_model(lstm_neurons=32, hidden_layers=2, lenth=30, dims=1, n_out=3):
    """# create_lstm_model

    Disc：
        ...
    """
    model = Sequential()
    with tf.variable_scope("hidden"):
        for i in range(hidden_layers - 1):
            model.add(LSTM(lstm_neurons, return_sequences=True, input_shape=(lenth,dims), kernel_initializer='uniform'))
            # model.add(BatchNormalization())  # like dropout
        model.add(LSTM(lstm_neurons, return_sequences=False, kernel_initializer='uniform')) 
        # model.add(BatchNormalization())  # like dropout
    model.add(Dense(n_out, kernel_initializer='uniform'))
    network_structure = "lstm_M" + str(lstm_neurons) + "_" + str(hidden_layers)
    return network_structure, model


def create_lstm_model_dropout(lstm_neurons=32, hidden_layers=2, lenth=30, dims=1, n_out=3, dropout_rate=0.5):
    """# create_lstm_model_dropout

    Disc：
        ...
    """
    model = Sequential()
    with tf.variable_scope("hidden"):
        for i in range(hidden_layers - 1):
            model.add(LSTM(lstm_neurons, return_sequences=True, input_shape=(lenth,dims), kernel_initializer='uniform'))
            model.add(Dropout(dropout_rate))
        model.add(LSTM(lstm_neurons, return_sequences=False, kernel_initializer='uniform')) 
        model.add(Dropout(dropout_rate))
    model.add(Dense(n_out, kernel_initializer='uniform'))
    network_structure = "lstm_M" + str(lstm_neurons) + "_" + str(hidden_layers)
    return network_structure, model


def create_bp_model(hidden_neurons=32, dims=1, n_out=3):
    """# create_bp_model

    Disc：
        ...
    """
    # create model
    model = Sequential()
    model.add(Dense(hidden_neurons, input_dim=dims, kernel_initializer='uniform'))
    model.add(Dense(hidden_neurons, kernel_initializer='uniform'))
    model.add(Dense(n_out, kernel_initializer='uniform'))  # notice: activation=sigmoid,(0,1)
    network_structure = "bp_M" + str(hidden_neurons)
    return network_structure, model


def create_rnn_model(hidden_neurons=32, lenth=30, dims=1, n_out=3):
    """# create_bp_model

    Disc：
        ...
    """
    # create model
    model = Sequential()
    model.add(SimpleRNN(hidden_neurons, return_sequences=True, input_shape=(lenth,dims), kernel_initializer='uniform'))
    model.add(SimpleRNN(hidden_neurons, return_sequences=False, kernel_initializer='uniform')) 
    model.add(Dense(n_out, kernel_initializer='uniform'))  # notice: activation=sigmoid,(0,1)
    network_structure = "rnn_M" + str(hidden_neurons)
    return network_structure, model


def run_mutiple_model(trainX, trainY, testX, testY, lenth=30, dims=1):
    """# run_mutiple_model

    Disc：
        ...
    """
    # 打开一个文件
    fo = open("experiments/lstm_univariate.txt", "w+")
    array_lstm_neurons = [16, 32, 64, 128]
    array_hidden_layers = [1, 2, 3, 4]
    # array_lstm_neurons = [16]
    # array_hidden_layers = [1, 2]
    
    for i in range(len(array_lstm_neurons)):
        for j in range(len(array_hidden_layers)):
            print(array_lstm_neurons[i], ",", array_hidden_layers[j])
            
            # create and fit the LSTM network
            network_structure, model = create_model(array_lstm_neurons[i], array_hidden_layers[j], lenth, dims)
            model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
            history = model.fit(trainX, trainY, epochs=50, batch_size=30, validation_split=0.25, verbose=0) #verbose：debug信息
            # epochs=30, nearly converge
             
            # make predictions
            trainPredict = model.predict(trainX)
            testPredict = model.predict(testX)
            # invert predictions
            trainPredict = scaler.inverse_transform(trainPredict)
            trainYInverse = scaler.inverse_transform([trainY])
            testPredict = scaler.inverse_transform(testPredict)
            testYInverse = scaler.inverse_transform([testY])
            
            # Plot training & validation loss values
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('Model loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='upper left')
            plt.savefig('experiments/loss_' + network_structure + '.png')
            
            fo.write('\n\n------------' + network_structure + "------------\n")
            # calculate mean absolute error
            trainScoreMAE = mean_absolute_error(trainYInverse[0], trainPredict[:,0])
            testScoreMAE = mean_absolute_error(testYInverse[0], testPredict[:,0])
            fo.write('\nTrain Score: %.2f MAE' % (trainScoreMAE))
            fo.write('\nTest Score: %.2f MAE' % (testScoreMAE))
            # calculate root mean squared error
            trainScoreRMSE = math.sqrt(mean_squared_error(trainYInverse[0], trainPredict[:,0]))
            testScoreRMSE = math.sqrt(mean_squared_error(testYInverse[0], testPredict[:,0]))
            fo.write('\nTrain Score: %.2f RMSE' % (trainScoreRMSE))
            fo.write('\nTest Score: %.2f RMSE' % (testScoreRMSE))
            # calculate R square
            trainScoreR = r2_score(trainYInverse[0], trainPredict[:,0])
            testScoreR = r2_score(testYInverse[0], testPredict[:,0])
            fo.write('\nTrain Score: %.2f R square' % (trainScoreR))
            fo.write('\nTest Score: %.2f R square ' % (testScoreR))
            
            keras.backend.clear_session()
            
    # 关闭打开的文件
    fo.close()
    return
    

