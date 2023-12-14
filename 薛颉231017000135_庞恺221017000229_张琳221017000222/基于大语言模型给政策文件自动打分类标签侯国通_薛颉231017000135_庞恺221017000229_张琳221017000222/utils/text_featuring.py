import pickle
import pandas as pd
import numpy as np
import torch as T
from torch.utils.data import Dataset, random_split

from config.params import PAD_NO, UNK_NO, START_NO, SENT_LENGTH

def load_file():
    with open('data/labels.pk', "rb") as f:
        labels = pickle.load(f)

    with open('data/chars.pk', "rb") as f:
        chars = pickle.load(f)

    label_dict = dict(zip(labels, range(len(labels))))
    char_dict = dict(zip(chars, range(len(chars))))
    return label_dict, char_dict

def text_feature(labels, contents, label_dict, char_dict):
    samples, y_true = [], []
    for s_label, s_content in zip(labels, contents):
        y_true.append(label_dict[s_label])
        train_sample = []
        for char in s_content:
            if char in char_dict:
                train_sample.append(START_NO + char_dict[char])
            else:
                train_sample.append(UNK_NO)
        # 补充或截断
        if len(train_sample) < SENT_LENGTH:
            samples.append(train_sample + ([PAD_NO] * (SENT_LENGTH - len(train_sample))))
        else:
            samples.append(train_sample[:SENT_LENGTH])

    return samples, y_true

def load_csv_file(file_path):
    print(file_path)
    df = pd.read_csv(file_path)
    samples, y_true = [], []
    for index, row in df.iterrows():
        y_true.append(row['label'])
        samples.append(row['content'])
    return samples, y_true


class CSVDataset(Dataset):
    # load the dataset
    def __init__(self, file_path):
        label_dict, char_dict = load_file()
        samples, y_true = load_csv_file(file_path)
        x, y = text_feature(y_true, samples, label_dict, char_dict)
        self.X = T.from_numpy(np.array(x)).long()
        self.y = T.from_numpy(np.array(y))

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

    # get indexes for train and test rows
    def get_splits(self, n_test=0.3):
        # determine sizes
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        # calculate the split
        return random_split(self, [train_size, test_size])
