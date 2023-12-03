import os
from io import open
import torch
import torch.nn as nn
import model
import time

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

device = torch.device("mps")

rnn_type = ["LSTM", "RNN_TANH", "RNN_RELU", "GRU"]
ntokens = len(idx2word)
emsize = 200     
nhid = 200
nlayers = 2
dropout = 0.2
tied = False

model = model.RNNModel(rnn_type[0], ntokens, emsize, nhid, nlayers, dropout, tied).to(device)

print(model)
print(model.ntoken, model.drop, model.encoder, model.rnn, model.decoder, model.rnn_type, model.nhid, model.nlayers)

criterion = nn.NLLLoss()


lr = 20
best_val_loss = None

for epoch in range(1, 5):
    epoch_start_time = time.time()
    model.train()
    total_loss = 0.
    start_time = time.time()
    model.init_hidden(20)
    for batch, i in enumerate(range(0, ids.size(0) - 1, 35)):
    print(epoch, epoch_start_time)
