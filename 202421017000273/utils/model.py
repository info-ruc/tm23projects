# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import torch
import torch.nn as nn
from torch import mm as matrix_m
from torch import cat as catencation
from torch import randn as torch_randn
from torch import mean as torch_mean
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.functional import sigmoid, softmax, log_softmax

'''
编码器, 其中 input_dim 是字典中词的总数. 注意隐层只能是偶数,
否则可能报错.
'''
class Encoder(nn.Module):

    def __init__(self, input_dim, embedding_dim, hidden_dim, 
                     num_layers=3, bidirectional=False):

        super(Encoder, self).__init__()

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # 这里我用了单层，如果使用双向 LSTM, 那么预设 hidden 应该除 2,
        # 否则计算量会因为 2 倍隐层而增大许多
        if bidirectional:
            self.hidden_dim //= 2

        # 词嵌入矩阵, 注意是线性的.
        self.embedding = nn.Embedding(self.input_dim, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, 
                                    num_layers=self.num_layers, batch_first=True, 
                                        bidirectional=self.bidirectional)

    '''
	注意, 当使用 batch 时, x 作为输入张量 (batch, sent_len, word_len) 应该
	使得 sent_len 均相同, 不足的地方用 index=0 填充. 且要求样例根据句子长短
	从大到小, 从上往下排序. 并给定一个尺度为 (batch_size, 1) 表示各个句子长度
	的向量 seq_lens.

		input: (batch_size, sentence_length, embedding_length)
		output: (batch_size, sentence_length, hidden_length)
 	'''
    def forward(self, x, seq_lens):
        # 得到连续词向量.
        embedded_x = self.embedding(x)

        # batch 打包和解包.
        packed_input = pack_padded_sequence(embedded_x, seq_lens, batch_first=True)
        lstm_out, (h_last, c_last) = self.lstm(packed_input, None)
        padded_output, _ = pad_packed_sequence(lstm_out)

        last_start = self.hidden_dim * (self.num_layers - 1)
        if self.bidirectional:
            last_start *= 2

        # 使用 batch_first. 隐层的大小是 num_layers * hidden.
        # lstm.transpose(0, 1).contiguous().view(batch_size, -1).
        return padded_output.transpose(0, 1), \
                       (h_last.transpose(0, 1).contiguous().view(x.size(0), -1)[:, last_start:],\
                            c_last.transpose(0, 1).contiguous().view(x.size(0), -1)[:, last_start:])


'''
解码器. 其中 lstm_hidden_size 应该和 Encoder 中 LSTM 的隐层维度一样.
'''
class Decoder(nn.Module):

    def __init__(self, lstm_hidden_size,
                     slot_embedding_size, slot_output_size,
                                           intent_output_size):

        super(Decoder, self).__init__()

        self.lstm_hidden_size = lstm_hidden_size
        self.slot_embedding_size = slot_embedding_size
        self.slot_output_size = slot_output_size
        self.intent_output_size = intent_output_size

        # 输入尺度: slot_embedding + encoder_hidden_dimension + attention_dimension.
        self.lstm_cell = nn.LSTMCell(slot_embedding_size + 3 * lstm_hidden_size, lstm_hidden_size)
        # 输出预测 slot 分布的全连接层.
        self.slot_output = nn.Linear(lstm_hidden_size, slot_output_size)
        # 输出预测 intent 分布的全连接层.
        self.intent_output = nn.Linear(lstm_hidden_size * 2, intent_output_size)

        # 注意力机制.
        self.attention = nn.Linear(lstm_hidden_size * 2, 1)

        # slot 的嵌入矩阵.
        self.slot_embedding = nn.Embedding(slot_output_size, slot_embedding_size)
        # 初始化零响应 slot 的参数.
        self.init_slot = nn.Parameter(torch_randn(1, slot_embedding_size), requires_grad=True)

    def forward(self, lstm_hiddens, encoder_hiddens, seq_lens):

        ret_slot_softmax = []
        ret_intent = []
        for batch_idx in range(0, encoder_hiddens.size(0)):
            prev_lstm_hidden = (lstm_hiddens[0][batch_idx:batch_idx+1, :],
                                            lstm_hiddens[1][batch_idx:batch_idx+1, :])
            prev_slot_embedding = self.init_slot

            slot_softmax = []
            context_vectors = []
            for word_idx in range(0, seq_lens[batch_idx]):
                curr_encoder_hideen = encoder_hiddens[batch_idx, word_idx:word_idx+1, :]

                # 重复 prev_lstm_hidden.
                repeat_lstm_hidden = catencation([prev_lstm_hidden[0]]*seq_lens[batch_idx], dim=0)
                # 将 repeat_lstm_hidden 和 curr_encoder_hidden 拼接起来.
                combined_attention_input = catencation([repeat_lstm_hidden, encoder_hiddens[batch_idx, :seq_lens[batch_idx], :]], dim=1)
                attention_param = self.attention(combined_attention_input).transpose(0, 1).contiguous()
                # 用 softmax 归一化.
                attention_param = softmax(attention_param)

                # 取出第 batch 个 Encoder 隐向量簇.
                curr_encoder_hiddens = encoder_hiddens[batch_idx, :seq_lens[batch_idx], :]
                # 计算 attention 向量.
                curr_attention = matrix_m(attention_param, curr_encoder_hiddens)
                context_vectors.append(curr_attention)

                # 合并 s_{i - 1}, y_{i - 1}, h_i, c_i]
                combined_lstm_input = catencation([prev_lstm_hidden[0], prev_slot_embedding,
                                                                   curr_encoder_hideen, curr_attention], dim=1)
                prev_lstm_hidden = self.lstm_cell(combined_lstm_input, prev_lstm_hidden)

                # 输出 slot 的分布.
                slot_output = self.slot_output(prev_lstm_hidden[0])

                # 记录 slot 的输出分布.
                slot_softmax.append(log_softmax(slot_output))
                # 更新 prev_slot_embedding.
                _, max_idx = softmax(slot_output).topk(1, dim=1)
                prev_slot_embedding = self.slot_embedding(max_idx).squeeze(0)

            # 将 slot_softmax 的值都拼接起来.
            ret_slot_softmax.append(catencation(slot_softmax, dim=0))	

            # 预测 intent, 先做池化操作.

            context_matrix = catencation(context_vectors, dim=0)
            combined_pooling_matrix = catencation([context_matrix, encoder_hiddens[batch_idx, :seq_lens[batch_idx], :]], dim=1)
            reduce_pooling = torch_mean(combined_pooling_matrix, dim=0)
            ret_intent.append(log_softmax(self.intent_output(reduce_pooling)))

        return ret_slot_softmax, torch.stack(ret_intent)
