#coding: UTF-8
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from dataset import Dataset, config

#模型定义
class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, number_layers, drop_rate, batch_size, bidirect):
        super(LSTMTagger,self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tagset_size = tagset_size
        self.num_layers = number_layers
        self.drop_rate = drop_rate
        self.batch_size = batch_size
        self.bidirect = bidirect

        self.word_embeddings = nn.Embedding(self.vocab_size,self.embedding_dim)
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_size = self.embedding_dim, hidden_size = self.hidden_dim, num_layers = self.num_layers, \
                            dropout = self.drop_rate, batch_first=True, bidirectional=self.bidirect)
        ## 如果batch_first为True，输入输出数据格式是(batch, seq_len, feature)
        ## 为False，输入输出数据格式是(seq_len, batch, feature)，
        self.dropout = nn.Dropout(self.drop_rate)

        if self.bidirect:
            print("双向LSTM")
            self.hidden2tag = nn.Linear(self.hidden_dim*2, self.tagset_size)
        else:
            print("单向LSTM")
            self.hidden2tag = nn.Linear(self.hidden_dim, self.tagset_size)# hidden to tag

    def forward(self, sentence, lens_batch, is_test=False):
        embeds = self.word_embeddings(sentence)# sentence 是词seq的索引

        if is_test==False:
            self.input_tensor = nn.utils.rnn.pack_padded_sequence(embeds, lens_batch, batch_first=True)
            lstm_out,self.hidden = self.lstm(self.input_tensor)
            lstm_out_pack,_ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

            x = self.dropout(lstm_out_pack)#add
            x = torch.tanh(x)#add
            tag_space = self.hidden2tag(lstm_out_pack)

        else:
            lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1))
            x = self.dropout(lstm_out)
            x = torch.tanh(x)
            tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))

        return tag_space

# 训练实例
class LSTM_Model(object):
    def __init__(self, args, lstmTagger = LSTMTagger, DataSet = Dataset, config = config):
        self.config = config
        self.args = args
        self.epoch = self.args.epoch
        self.batch_size = self.config.BATCH_SIZE

        self.dataset = DataSet(self.config)
        self.training_data = self.dataset.training_data
        self.tag_weights = self.dataset.tag_weights
        self.test_data, self.test_tags = self.dataset.gene_testdata()
        self.word_to_ix_length = self.dataset.word_to_ix_length #词表长度

        self.max_accuracy = 0.

        if args.weighted_tag: #是否对类别加权
            self.tag_weights = torch.tensor(self.tag_weights,dtype=torch.float)
            self.loss_function = nn.CrossEntropyLoss(weight=self.tag_weights)
        else:
            self.loss_function = nn.CrossEntropyLoss()

        self.lstm_model = lstmTagger(self.config.EMBEDDING_DIM, self.config.HIDDEN_DIM, self.word_to_ix_length,\
                          len(self.config.TAG_to_ix), self.config.LAYER, self.config.DROP_RATE, self.config.BATCH_SIZE,\
                          self.args.bidirection)
       
    #模型训练
    def train(self):
        self.lstm_model.train() # set mode
        self.optimizer = optim.SGD(self.lstm_model.parameters(), lr=self.config.LR, weight_decay=self.config.weight_decay)

        self.lstm_model.cuda(0)
        self.loss_function.cuda(0)

        for epoch in range(self.epoch):
            print("\n================= Epoch:",epoch,"/",self.epoch,"===============")
            start = time.perf_counter()
            _iter,loss_total = 0,0

            while _iter < len(self.training_data):
                #try:
                    sentences_batch,tags_batch_list,lens_batch = self.dataset.gene_batch(self.training_data, _iter) #（tensor 形式的sentens,tags）
                    sentences_batch = sentences_batch.cuda(0)
                    for i in range(len(tags_batch_list)):
                        tags_batch_list[i] = torch.tensor(tags_batch_list[i], dtype=torch.long)
                        tags_batch_list[i] = tags_batch_list[i].cuda(0)

                    self.optimizer.zero_grad()
                    
                    tag_scores = self.lstm_model(sentences_batch, lens_batch)

                    # tags_batch_list = batch_size * 词个数 (对应词性的索引) # tag_scores = batch_size * 词个数 * 26类
                    loss = 0
                    for i in range(min(self.batch_size,len(tags_batch_list))):
                        loss += self.loss_function(tag_scores[i][:lens_batch[i]], tags_batch_list[i])
                    loss_total += loss

                    if (_iter + self.batch_size) % (100 * self.batch_size) == 0:
                        print("iter ",_iter, "loss:",loss_total/100, " lr:",self.optimizer.state_dict()['param_groups'][0]['lr'])
                        loss_total = 0
                        
                    _iter += self.batch_size
                    loss.backward()
                    self.optimizer.step()
                #except:
                #    print(item)
                #    continue

            # Test
            self.test()
            print("One epoch use time:",time.perf_counter()-start)

    def test(self):
        print("******Testing...")
        self.lstm_model.cuda(0)
        with torch.no_grad(): # test mode
            num = 0
            total_word = 0
            test_labels = []
            test_predicts = []
            for inputs, targets in zip(self.test_data[:3850],self.test_tags[:3850]):
                #try:
                    total_word += len(inputs) #测试集的所有词数量
                    inputs = torch.tensor(inputs, dtype=torch.long)
                    targets = torch.tensor(targets, dtype=torch.long)
                
                    inputs = inputs.cuda(0)
                    targets = targets.cuda(0)

                    tag_scores = self.lstm_model(inputs,[0],is_test=True) # tensor N个词*tag数
                    tag_scores_numpy = tag_scores.cpu().numpy()
                
                    for idx,word in enumerate(tag_scores_numpy):
                        test_tag = np.where(word == np.max(word))
                        if test_tag[0][0] == int(targets[idx]):
                            num += 1
                        test_labels.append(int(targets[idx]))
                        test_predicts.append(test_tag[0][0])
                #except:
                #    print(item)
                #    continue

            # 评测
            accuracy = num / total_word
            if accuracy > self.max_accuracy:
                self.max_accuracy = accuracy
                # 计算混淆矩阵
                self.Confusion_matrix(test_labels, test_predicts)
                # save model
                torch.save(self.lstm_model.state_dict(),self.args.checkpoint+'/epoch_max_accuracy.pkl')
                print("Max accuracy's model is saved in ",self.args.checkpoint+'/epoch_max_accuracy.pkl')
            
            print("Acc:",accuracy,"(",num,'/',total_word,")","   Max acc so far:",self.max_accuracy) #准确率
            
    def Confusion_matrix(self, test_labels, test_predicts):
        label_list = self.dataset.TAG_list_test
        cm = confusion_matrix(test_labels, test_predicts)
        cm = cm.astype(np.float32)
        sums = []
        for i in range(len(label_list)-1):#'x'类别没有
            sums.append(np.sum(cm[i]))
        
        for i in range(len(sums)):
            for j in range(len(sums)):
                cm[i][j]=round(float(cm[i][j])/float(sums[i]),2)

        np.savetxt(self.args.checkpoint+'/Con_Matrix.txt', cm, fmt="%.2f", delimiter=',') #保存为2位小数的浮点数，用逗号分隔
        print("The confusion matrix is saved in "+self.args.checkpoint+"/Con_Matrix.txt")

