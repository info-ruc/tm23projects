'''
predict from lastest data, insert into prediction table
'''
import csv

from numpy import array
import pandas as pd
from keras.models import load_model
from datetime import datetime


# insert the prediction
def insert_prediction(date, prediction):
    with open('data/prediction.csv', 'a',) as csvfile:
        print("date", date, "prediction", prediction)
        csvfile.write(f'{date},{prediction}\n')
        try:
            csvfile.write(f'{date},{prediction}\n')
        except Exception as err:
            print(f'{date} {prediction}: {err.args}')

    with open('data/prediction.csv', 'r') as file:
        lines = file.readlines()
        last_line = lines[-1].strip()
        last_date, last_prediction = last_line.split(',')
        if last_date == date and last_prediction == prediction:
            print("Data was written successfully.")
        else:
            print("Failed to write data.")

# predict from saved model
def predict(model, predict_data_list):
    n_steps = 1
    n_features = 3

    # demonstrate prediction    
    x_input = array(predict_data_list)
    x_input = x_input.reshape((1, n_steps, n_features))
    yhat = model.predict(x_input, verbose=0)

    answer = yhat.flatten().tolist()
    ans = ''.join(str(x) for x in answer)
    return ans


def get_unpredicted_data():
    data_row = pd.read_csv('data/historical_data.csv')
    return data_row


def main():
    model = load_model('model/7th_days.h5')
    now = datetime.utcnow().strftime('%Y-%m-%d %H:%M')

    data_row = get_unpredicted_data()

    date_list = list(data_row['future_date'])
    bitcoin_list = list(data_row['bitcoin_price'])
    gold_list = list(data_row['gold_price'])
    oil_list = list(data_row['oil_price'])

    for i in range(len(date_list)):
        predict_data_list = [bitcoin_list[i], gold_list[i], oil_list[i]]
        prediction = predict(model=model, predict_data_list=predict_data_list)
        insert_prediction(date=date_list[i], prediction=prediction)

    print(f'{now}: Predict Done')


if __name__ == '__main__':
    main()

# %%
