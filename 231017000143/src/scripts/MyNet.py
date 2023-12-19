from torch import nn

class MyNet(nn.Module):
    def __init__(self, vocab_size, embeding, hiddens, lstm_layers):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embeding)
        self.lstm = nn.LSTM(embeding, hiddens, lstm_layers, batch_first=True)
        self.h2h = nn.Linear(hiddens, hiddens)
        self.h2o = nn.Linear(hiddens, vocab_size)

    def forward(self, word_ids, lstm_hidden=None):
        embedded = self.embedding(word_ids)
        lstm_out, lstm_hidden = self.lstm(embedded, lstm_hidden)
        out = self.h2h(lstm_out)
        out = self.h2o(out)

        return out, lstm_hidden