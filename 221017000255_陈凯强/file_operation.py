import pandas as pd
import numpy as np
from torch.utils.data import Dataset, random_split
from common import PAD_NO, UNK_NO, START_NO, SENT_LENGTH, NUM_WORDS, EMBEDDING_SIZE, LABELS_FILE_PATH, CHARS_FILE_PATH
import pickle

# 文件操作
def savePkFile(file_path, data):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def readPkFile(file_path):
    with open(file_path, "rb") as f:
        content = pickle.load(f)
    return content

# load csv file
def load_csv_file(file_path):
    df = pd.read_csv(file_path)
    samples, y_true = [], []
    for index, row in df.iterrows():
        y_true.append(row['label'])
        samples.append(row['content'])
    return samples, y_true


# 读取pickle文件
def load_pickle_file():
    labels = readPkFile(file_path=LABELS_FILE_PATH)
    chars = readPkFile(file_path=CHARS_FILE_PATH)
    label_dict = dict(zip(labels, range(len(labels))))
    char_dict = dict(zip(chars, range(len(chars))))
    return label_dict, char_dict


# 文本预处理
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


# dataset
class CSVDataset(Dataset):
    # load the dataset
    def __init__(self, file_path):
        label_dict, char_dict = load_pickle_file()
        samples, y_true = load_csv_file(file_path)
        x, y = text_feature(y_true, samples, label_dict, char_dict)
        self.X = torch.from_numpy(np.array(x)).long()
        self.y = torch.from_numpy(np.array(y))

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

import torch
from gensim.models import KeyedVectors
from file_operation import load_pickle_file


def getPretrainedVector():
    # 读取转换后的文件
    label_dict, char_dict = load_pickle_file()
    # 加载转化后的文件
    model = KeyedVectors.load_word2vec_format('./pretrain_vector/sgns.wiki.char.bz2',
                                              binary=False,
                                              encoding="utf-8",
                                              unicode_errors="ignore")
    # 使用gensim载入word2vec词向量
    pretrained_vector = torch.zeros(NUM_WORDS + 4, EMBEDDING_SIZE).float()
    # print(model.index2word)
    for char, index in char_dict.items():
        if char in model.vocab:
            # vector = model.get_vecattr(char)
            vector = model.get_vector(char)
            # print(vector)
            pretrained_vector[index, :] = torch.from_numpy(vector)
    return pretrained_vector