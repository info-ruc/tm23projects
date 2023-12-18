import math
import torch
import torch.nn as nn

from common import NUM_WORDS, EMBEDDING_SIZE
from file_operation import getPretrainedVector


# Transformer位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, vocab_size=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(vocab_size, d_model)
        position = torch.arange(0, vocab_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# 基于pytorch TransformerEncoder的文本分类器
class TextClassifier(nn.Module):
    def __init__(
            self,
            nhead: int = 8,
            dim_feedforward: int = 2048,
            nlayers: int = 6,
            dropout: float = 0.1
    ):
        super().__init__()
        self.model_type = 'Transformer'
        vocab_size = NUM_WORDS + 2
        d_model = EMBEDDING_SIZE
        # vocab_size, d_model = embeddings.size()
        assert d_model % nhead == 0, "nheads must divide evenly into d_model"

        # Embedding layer definition
        # self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        pretrained_vector = getPretrainedVector()
        self.emb = nn.Embedding.from_pretrained(pretrained_vector, freeze=False, padding_idx=0)

        self.pos_encoder = PositionalEncoding(
            d_model=d_model,
            dropout=dropout,
            vocab_size=vocab_size
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=nlayers
        )
        self.classifier = nn.Linear(d_model, 5)
        self.d_model = d_model

    def forward(self, x):
        x = self.emb(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x
