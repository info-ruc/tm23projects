#encoding:gbk

import torch
from torch.utils.data import DataLoader
from numpy import vstack, argmax
from sklearn.metrics import classification_report

from utils.text_featuring import CSVDataset
from config.params import TEST_BATCH_SIZE, TEST_FILE_PATH, MODEL_PATH

if __name__ == '__main__':
    model = torch.load(MODEL_PATH).eval()
    label_name = {'科技', '民生', '经济', '其他'}
    predictions = []
    groundTruth = []
    test = CSVDataset(TEST_FILE_PATH)
    test_dl = DataLoader(test, batch_size=TEST_BATCH_SIZE)
    for i, (inputs, targets) in enumerate(test_dl):
        yhat = model(inputs)
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        yhat = argmax(yhat, axis=1)
        for j in range(len(yhat)):
            predictions.append(yhat[j])
            groundTruth.append(actual[j])
    
    print(classification_report(groundTruth, predictions, target_names=label_name))