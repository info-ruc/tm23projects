import os
from io import open
import torch

word2idx = {}
idx2word = []

with open("../dataset/test-slim.txt", 'r', encoding="utf8") as f:
    for line in f:
        words = line.split() + ['<eos>']
        for word in words:
            if word not in word2idx:
                idx2word.append(word)
                word2idx[word] = len(idx2word) - 1

with open("../dataset/test-slim.txt", 'r', encoding="utf8") as f:
    idss = []
    for line in f:
        words = line.split() + ['<eos>']
        ids = []
        for word in words:
            ids.append(word2idx[word])  
        idss.append(torch.tensor(ids).type(torch.int64))
    ids = torch.cat(idss)
    print(idss)      
    print(type(ids))      
    print(ids.size())      
    print(ids.dtype)      
    print(ids.numel())      
