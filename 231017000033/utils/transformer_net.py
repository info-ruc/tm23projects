import torch
import math
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# [位置编码器] 在Transformer的编码器结构中，没有针对词汇位置信息的处理，因此在Embedding层加入位置编码器
# 其将词汇位置的语义信息加入到词嵌入张量中，以弥补位置信息的缺失
class PositionalEncoding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, dropout: float = 0.1):
        # vocab_size: 词汇表大小
        # embed_dim: 词嵌入维度
        # dropout : 置零比率
        # 最终输出一个加入了位置编码信息的词嵌入张量
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # 初始化位置编码矩阵，大小为(vocab_size, embed_dim)，值全为0
        pe = torch.zeros(vocab_size, 1, embed_dim)
        # 初始化绝对位置矩阵，大小为(vocab_size, 1), 用词汇的索引表示它的绝对位置。
        # 先用tensor.arange方法获得一个连续自然数向量, 然后用unsqueeze拓展向量维度
        # 又因参数传递的是1，代表矩阵拓展的位置，会使向量变成一个的矩阵
        #position = torch.arange(0, vocab_size, dtype=torch.float).unsqueeze(1);
        position = torch.arange(vocab_size).unsqueeze(1)
        # 绝对位置矩阵初始化后，考虑将位置信息加入到位置编码矩阵中。
        # 由于position大小为(vocab_size, 1)，而pe大小为(vocab_size, embed_dim)，则需position乘以
        # 大小为(1, embed_dim)的矩阵，即div_term，同时希望它能够将自然数的绝对位置编码缩放成足够小的数字，
        # 以便于在之后的梯度下降过程中更快收敛
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        # 用sin和cos交替来表示token的位置
        #pe[:, 0::2] = torch.sin(position * div_term)
        #pe[:, 1::2] = torch.cos(position * div_term)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        #pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    # 将词的嵌入向量与位置编码相加作为self-attention模块的输入
    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        # x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)

# vocab_size：词汇表大小
# embed_dim：词嵌入向量维度
# num_class：分类数
# depth： transformer encoder层数
class TransformerNet(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_class: int, num_heads: int, depth: int, dropout: float = 0.5):
        super().__init__()
        # 位置编码器
        self.pos_encoder = PositionalEncoding(vocab_size, embed_dim, dropout)
        # encoder层
        encoder_layers = TransformerEncoderLayer(embed_dim, num_heads, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, depth)
        # 词嵌入层
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.embed_dim = embed_dim
        # 线性神经网络层
        self.linear = nn.Linear(embed_dim, num_class)
        # 初始化权重
        self.init_weights()

    # 使用[-0.1, 0.1]范围的随机数初始化权重
    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    # 前向传播函数
    def forward(self, x, offsets):
        # 词嵌入
        x = self.embedding(x, offsets) * math.sqrt(self.embed_dim)
        # 编码位置
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        return self.linear(x)