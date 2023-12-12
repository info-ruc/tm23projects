# Price prediction based on text
This is a simple price prediction model based on text. The model uses a bag of words approach to extract features from the text and then trains a linear regression model to predict the price. The model is trained on the dataset from Kaggle. The model is deployed using Flask.

## Dataset
The dataset used for this project is the [Airbnb New York City listings](https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data)

## Model
The model is trained using BERT. The model is trained on the text data and the price data. The model is trained using the following steps:

1. Load the data
2. Clean the data
3. Tokenize the data
5. Convert the data into tensors
6. Create the model
7. Train the model
8. Evaluate the model
9. Save the model

## Usage
To use the model, you need to install the required libraries. You can do this by running the following command in your terminal:

```bash
conda env create -f freeze.yml # install
```

After installing the libraries, you can run the following command to start the Flask server:

```bash
python train.py
```