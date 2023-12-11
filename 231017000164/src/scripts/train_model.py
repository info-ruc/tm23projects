# train a model to predict the next 7th day

# multivariate lstm example
from numpy import array
from numpy import hstack
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import pandas as pd


# split a multivariate sequence into samples
def split_sequences(dataset, n_steps):
	X, y = list(), list()

	for i in range(len(dataset)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix > len(dataset):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = dataset[i:end_ix, :-1], dataset[end_ix-1, -1]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)


def train_model(dataset, n_steps):
    # convert into input/output
    X, y = split_sequences(dataset=dataset, n_steps=n_steps)

    # the dataset knows the number of features, e.g. 2
    print("X", X)
    n_features = X.shape[2]

    # define model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # fit model
    model.fit(X, y, epochs=200, verbose=0)

    return model, n_features


# generate training dataset from raw data
def prepare_data(raw_data):
    # define input sequence
    bitcoin_in = array(list(raw_data['bitcoin_price']))
    gold_in = array(list(raw_data['gold_price']))
    oil_in = array(list(raw_data['oil_price']))

    # output seqquence    
    future_seq = array(list(raw_data['future_price']))

    # convert to [rows, columns] structure
    bitcoin_in = bitcoin_in.reshape((len(bitcoin_in), 1))
    gold_in = gold_in.reshape((len(gold_in), 1))
    oil_in = oil_in.reshape((len(oil_in), 1))
    future_seq = future_seq.reshape((len(future_seq), 1))

    # horizontally stack columns
    dataset = hstack((bitcoin_in, gold_in, oil_in, future_seq))

    return dataset


# generate training raw data
def get_dataset():
    raw_data = pd.read_csv('data/historical_data.csv')
    return raw_data


def main(start_date_str, end_date_str):
    # connection, cursor = connect_db()
    raw_data = get_dataset()
    dataset = prepare_data(raw_data=raw_data)

    # choose a number of time steps
    n_steps = 1
    print("dataset", dataset, "n_steps", n_steps)
    model, n_features = train_model(dataset=dataset, n_steps=n_steps)

    model.save('model/7th_days.h5')

    # demonstrate prediction
    # 2023/12/07 bitcoin_price gold_price oil_price: 43788.2, 2018.79, 64.76
    x_input = array([43788.2, 2018.7, 64.76])
    x_input = x_input.reshape((1, n_steps, n_features))
    yhat = model.predict(x_input, verbose=0)
    
    # prediction 2023/12/7，real_price 43671.1
    print(yhat)


if __name__ == '__main__':
    '''
    start date、end date training dataset period, format: 2014-09-17
    '''
    start_date = '2014-09-17'
    end_date = '2023-12-07'
    main(start_date_str=start_date, end_date_str=end_date)
#%%
