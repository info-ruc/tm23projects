import numpy as np
import torch

# 超参数
class Config:
    TAG_to_ix = {"n":0,"t":1,"s":2,"f":3,"m":4,"q":5,"b":6,"r":7,"v":8,"a":9,"z":10,"d":11,"p":12,"c":13,\
            "u":14,"y":15,"e":16,"o":17,"i":18,"l":19,"j":20,"h":21,"k":22,"g":23,"x":24,"w":25} # 空的补0
    TAG_to_ix_crf = {"n":0,"t":1,"s":2,"f":3,"m":4,"q":5,"b":6,"r":7,"v":8,"a":9,"z":10,"d":11,"p":12,"c":13,\
            "u":14,"y":15,"e":16,"o":17,"i":18,"l":19,"j":20,"h":21,"k":22,"g":23,"x":24,"w":25,"<START>":26,"<STOP>":27}

    FILE_NAME = "./data/1998-01-2003_shuf.txt"

    EMBEDDING_DIM = 100#6 # 词向量的特征维度
    HIDDEN_DIM = 200 # 隐藏层的特征维度
    LAYER = 1 #LSTM层数
    MAX_len = 660

    DROP_RATE = 0.5 #Dropout概率
    BATCH_SIZE = 24 # 批大小
    
    LR = 0.015
    weight_decay = 1.0e-8
    gamma = 0.98
    step = 200
    

config = Config()

#人民日报语料
class Dataset(object):
    def __init__(self, config=config):
        self.config = config
        self.TAG_list = []
        self.TAG_list_test = []
        for i in range(len(self.config.TAG_to_ix)):
            self.TAG_list.append(0)
            self.TAG_list_test.append(0)

        self.batch_size = self.config.BATCH_SIZE
        self.training_data, self.tag_weights = self.get_training_set()
        self.testing_data = self.get_testing_set()
        self.word_to_ix = self.get_word_to_ix()
        self.word_to_ix_length = len(self.word_to_ix) #词表长度

    def get_training_set(self): 
        '''
        return training data: list[ word_seq, tag_seq ] 原始数据，未经过编码
               tag_weights: 各tag类别权重
        '''
        training_data = []
        print(self.config.FILE_NAME)
        self.lines = open(self.config.FILE_NAME).readlines()
        for line in self.lines[:int(len(self.lines)*0.8)]:
            if not len(line.strip()): # 跳过空行
                continue
            word_str = ' '
            tag_list = ' '
            for item in line.strip().split()[1:]:
                if item.split('/')[0][0]=='[':
                    word_str += item.split('/')[0][1:]+' '
                else:
                    word_str += item.split('/')[0]+' '
                if item.split('/')[1][0].isupper(): #大写
                    tag_now = item.split('/')[1][1]
                    tag_list += tag_now +' '
                else:
                    tag_now = item.split('/')[1][0]
                    tag_list += tag_now+' '
                
                list_keys= [ i for i in self.config.TAG_to_ix.keys()]
                self.TAG_list[list_keys.index(tag_now)] += 1

            training_data.append((word_str.split(), tag_list.split()))
        
        tag_numpy = np.array(self.TAG_list,dtype=np.float32)
        tag_weights = (tag_numpy+1) / np.sum(tag_numpy) # +1是为防止某一类没有数据
        tag_weights = np.median(tag_weights)/tag_weights
        
        return training_data,tag_weights

    def get_testing_set(self):
        '''
        return testing data: list[ word_seq, tag_seq ] 原始数据，未经过编码
        '''
        testing_data = []
        self.lines = open(self.config.FILE_NAME).readlines()
        for line in self.lines[int(len(self.lines)*0.8):]:
            if not len(line.strip()):
                continue
            word_str = ' '
            tag_list = ' '
            for item in line.strip().split()[1:]:
                if item.split('/')[0][0]=='[':
                   word_str += item.split('/')[0][1:]+' '
                else:
                    word_str += item.split('/')[0]+' '

                if item.split('/')[1][0].isupper(): #大写
                    tag_now = item.split('/')[1][1]
                    tag_list += tag_now +' '
                else:
                    tag_now = item.split('/')[1][0]
                    tag_list += tag_now+' '
                list_keys= [ i for i in self.config.TAG_to_ix.keys()]
                self.TAG_list_test[list_keys.index(tag_now)] += 1
           
            testing_data.append((word_str.split(), tag_list.split()))

        return testing_data

    def get_word_to_ix(self):
        '''
        return self.word_to_ix: 为所有词赋予编号，词表
        '''
        word_to_ix = {}
        for sent,tag in self.training_data + self.testing_data:
            for word in sent:
                if word not in word_to_ix:#防止重复记录单词
                    word_to_ix[word] = len(word_to_ix)+1 #空的情况是补0
        #print('word table length:',len(word_to_ix))
        return word_to_ix # 56182 

    def prepare_sequence(self, seq, to_ix): 
        '''
        # 通过查询词表to_ix, 返回seq中每个词的索引，返回list形式
        '''
        idxs = [to_ix[w] for w in seq]
        return idxs

    def gene_batch(self, data, _iter):
        '''
        traindata loader for training;
        对不同长度的句子进行补齐，便于batch-based train
        '''
        batch_i = 0  #句子长度不同！！！
        train_batch_sentences = []
        train_batch_tags = []
        train_len = []
        for sentence, tag in data[_iter:]:
            train_len.append(len(sentence))

            pad_sentence = self.prepare_sequence(sentence, self.word_to_ix)
            pad_tag = self.prepare_sequence(tag, self.config.TAG_to_ix)

            # padding word_seq to the SAME len
            while len(pad_sentence)!=self.config.MAX_len:
                pad_sentence.append(0)

            # get a batch
            train_batch_sentences.append(pad_sentence)
            train_batch_tags.append(pad_tag)

            batch_i += 1
            if batch_i == self.batch_size:
                break   

        # 对batch中所有word_seq按长度降序排列
        down_index = np.array(train_len).argsort()
        down_index_list = down_index.tolist()
        down_index_list.reverse() 
        
        train_batch_sentences_down = []
        train_batch_tags_down = []
        train_len_down = []

        for i in range(min(self.batch_size,len(train_batch_sentences))):#防止最后一个不足batch
            train_batch_sentences_down.append(train_batch_sentences[down_index_list[i]])
            train_batch_tags_down.append(train_batch_tags[down_index_list[i]])
            train_len_down.append(train_len[down_index_list[i]])

        # 已经按长度降序排列
        tensor_sentences = torch.tensor(train_batch_sentences_down, dtype = torch.long)
        # tags 长度不一致，所以不可以直接tensor
        #tensor_tags = torch.tensor(train_batch_tags_down, dtype = torch.long)
        
        return (tensor_sentences, train_batch_tags_down, train_len_down)

    def gene_traindata(self):
        '''
        testdata loader for testing;
        不基于batch了，不必对齐
        '''
        inputs_set = []
        targets_set = []
        for item in self.training_data:
            inputs = self.prepare_sequence(item[0],self.word_to_ix)
            targets = self.prepare_sequence(item[1],self.config.TAG_to_ix)

            inputs_set.append(inputs)
            targets_set.append(targets)
        
        return inputs_set, targets_set
        
    def gene_testdata(self):
        '''
        testdata loader for testing;
        不基于batch了，不必对齐
        '''
        inputs_set = []
        targets_set = []
        for item in self.testing_data:
            inputs = self.prepare_sequence(item[0],self.word_to_ix)
            targets = self.prepare_sequence(item[1],self.config.TAG_to_ix)

            inputs_set.append(inputs)
            targets_set.append(targets)
        
        return inputs_set, targets_set



