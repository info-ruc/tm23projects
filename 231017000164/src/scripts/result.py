'''
insert verifiable prediction to result table
'''
from datetime import datetime
import pandas as pd


# insert verifiable prediction to result table
def insert_result_table():

    try:
        # read historical_data
        historical_data = pd.read_csv('data/historical_data.csv', usecols=['date', 'bitcoin_price'])

        # read prediction
        prediction = pd.read_csv('data/prediction.csv', usecols=['date', 'prediction'])
        historical_data['date'] = pd.to_datetime(historical_data['date'])
        prediction['date'] = pd.to_datetime(prediction['date'])

        historical_data['date'] = historical_data['date'].dt.strftime('%Y/%m/%d')
        prediction['date'] = prediction['date'].dt.strftime('%Y/%m/%d')

        merged_df = pd.merge(historical_data, prediction, on='date', how='inner')
        # bitcoin_price real_price
        merged_df = merged_df.rename(columns={'bitcoin_price': 'real_price'})
        merged_df = merged_df.reindex(columns=['date', 'prediction', 'real_price'])
        merged_df = merged_df.drop_duplicates(subset=['date'])
        merged_df.to_csv('data/result.csv', index=False)

        print("merge done.")

    except Exception as err:
        print(err.args)


def main():
    now = datetime.utcnow().strftime('%Y-%m-%d %H:%M')
    insert_result_table()

    print(f'{now}: Insert Result Done')


if __name__ == '__main__':
    main()

# %%
