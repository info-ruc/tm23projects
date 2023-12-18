import numpy as np
import torch
from torch.utils.data import  Dataset

class MyDataset(Dataset):
    def __init__(self, seqs, file="../dataset/poetry_trains.txt"):
        SOS = 0
        EOS = 1

        self.seqs = seqs
        with open(file, encoding="utf-8") as f:
            lines = f.read().splitlines()

        self.word2index = {"<SOS>": SOS, "<EOS>": EOS}
        indices = []
        num_words = 0
        for line in lines:
            indices.append(SOS)
            for word in line:
                if word not in self.word2index:
                    self.word2index[word] = num_words
                    num_words += 1
                indices.append(self.word2index[word])
            indices.append(EOS)

        self.index2word = {v: k for k, v in self.word2index.items()}
        self.data = np.array(indices, dtype=np.int64)

    def __len__(self):
        return (len(self.data) - 1) // self.seqs

    def __getitem__(self, i):
        start = i * self.seqs
        end = start + self.seqs
        return (
            torch.as_tensor(self.data[start:end]),
            torch.as_tensor(self.data[start + 1 : end + 1]),
        )
