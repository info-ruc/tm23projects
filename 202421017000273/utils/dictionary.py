# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from numpy import argsort
from numpy.random import shuffle
from numpy.random import seed

'''
储存以单词 word 或标注 slot 或目的 intent
的字典. 支持添加, 查询等操作.
'''


class Alphabet(object):

    def __init__(self, name="new_alphabet"):

        self.name = name

        # 储存所有单词的列表.
        self.word_list = []
        # 映射 : word -> index.
        self.word2index = {}
        # 映射 : index -> word.
        self.index2word = {}

    '''
    添加单个单词或者一个列表的单词.
    '''

    def add_words(self, elems):
        if isinstance(elems, list):
            for elem in elems:
                self.add_words(elem)
        elif isinstance(elems, str):
            # 注意字典中的元素不能重复加入, 重复者被忽略.
            if elems not in self.word_list:
                self.word2index[elems] = len(self.word_list) + 1
                self.index2word[self.word2index[elems]] = elems
                self.word_list.append(elems)
        else:
            raise Exception("加入元素必须是字符串或者字符串的列表.")

    '''
    返回所有单词的列表.
    '''

    def get_words(self):
        return self.word_list

    '''
    返回两个字典:
        1, word2index: word -> index
        2, index2word: index -> word
    '''

    def get_dicts(self):
        return self.word2index, self.index2word

    '''
     查询单个单词或者一个列表的单词.
    '''

    def indexs(self, words):
        if isinstance(words, list):
            ret_list = []
            for word in words:
                ret_list.append(self.indexs(word))

            return ret_list
        elif isinstance(words, str):
            if words not in self.word2index.keys():
                raise Exception("查询元素不在字典中.")
            else:
                return self.word2index[words]
        # 否则都不是抛出错误.
        else:
            raise Exception("查询元素必须是字符串或者字符串的列表.")

    '''
    查询单个序号对应的单词或者一个列表的序列.
    '''

    def words(self, idxs):
        if isinstance(idxs, list):
            ret_list = []
            for index in idxs:
                ret_list.append(self.words(index))

            return ret_list
        elif idxs == 0.0:
            return '<padding>'
        elif isinstance(idxs, int):
            return self.index2word[idxs]
        else:
            raise Exception("查询元素必须是整形 Int 或者整形的列表.")

    '''
    返回字典的名字, 它与保存数据的路径有关.
    '''

    def get_name(self):
        return self.name

    '''
    将数据保存到指定路径下, 有默认值.
    '''

    def save(self, file_dir):
        self.write_list(self.word_list, file_dir + self.name + '-word_list.txt')
        self.write_dict(self.word2index, file_dir + self.name + '-word2index.txt')
        self.write_dict(self.index2word, file_dir + self.name + '-index2word.txt')

    '''
     加载已缓存在硬盘上的数据到对象.
    '''

    def load(self, file_dir='./save/alphabets/'):
        self.word_list = self.read_list(file_dir + self.name + '-word_list.txt')
        self.word2index = self.read_dict(file_dir + self.name + '-word2index.txt')
        self.index2word = self.read_dict(file_dir + self.name + '-index2word.txt')

    '''
    相当于 Java 中对象 Object 的 toString 方法.
    '''

    def __str__(self):
        return "元素字典 " + self.name + " 包含以下元素: \n" + str(self.word_list) + \
            "\n\n其中映射 元素 -> 序号 如下: \n" + str(self.word2index) + \
            "\n\n其中映射 序列 -> 元素 如下: \n" + str(self.index2word) + '\n'

    '''
    返回字典中词的总数.
    '''

    def __len__(self):
        return len(self.word_list)

    '''
    读写文件的辅助函数.
    '''

    def write_list(self, w_list, file_path):
        with open(file_path, 'w') as fr:
            for word in w_list:
                fr.write(word + '\n')

    '''
    读写文件的辅助函数.
    '''

    def read_list(self, file_path):
        ret_list = []
        with open(file_path, 'r') as fr:
            for line in fr.readlines():
                ret_list.append(line.strip())

        return ret_list

    '''
    读写文件的辅助函数.
    '''

    def write_dict(self, dictionary, file_path):
        with open(file_path, 'w') as fr:
            for pair in dictionary.items():
                fr.write(str(pair[0]) + '\t' + str(pair[1]) + '\n')

    '''
    读写文件的辅助函数.
    '''

    def read_dict(self, file_path):
        ret_dict = {}
        with open(file_path, 'r') as fr:
            for line in fr.readlines():
                items = line.strip().split()
                try:
                    ret_dict[int(items[0])] = items[1]
                except Exception:
                    ret_dict[items[0]] = int(items[1])

        return ret_dict


class Dataset(object):

    def __init__(self, name="dataset",
                 word_alphabet=None, label_alphabet=None, intent_alphabet=None,
                 train_set=None, test_set=None, dev_set=None,
                 random_state=0):

        self.name = name

        self.word_alphabet = word_alphabet
        self.label_alphabet = label_alphabet
        self.intent_alphabet = intent_alphabet

        self.train_set = train_set
        self.test_set = test_set
        self.dev_set = dev_set

        # 将训练集随机化.
        self.random_state = random_state

    '''
    返回 alphabets.
    '''

    def get_alphabets(self):
        return self.word_alphabet, self.label_alphabet, self.intent_alphabet

    '''
    type: ['train', 'test']
    返回训练集(测试集).
    '''

    def get_dataset(self, type_):
        assert type_ in ['train', 'test', 'dev']

        if type_ == 'train':
            return self.train_set
        elif type_ == 'test':
            return self.test_set
        else:
            return self.dev_set

    '''
    读取文件数据, 其数据格式要求文件中每行格式:

    BOS word 1 word 2 ... word n EOS tag 1 tag 2 ... tag n intent

    '''

    def read_data(self, file_path):
        sentence_list = []
        labels_list = []
        intent_list = []

        with open(file_path, 'r') as fr:
            for line in fr.readlines():
                items = line.strip().split('EOS')

                sentence_list.append(items[0].split()[1:])
                labels_list.append(items[1].split()[1:-1])
                intent_list.append(items[1].split()[-1])

        return sentence_list, labels_list, intent_list

    '''
    搭建 word, label 和 intent 的字典, 注意这个 file_path 是 name.all.txt
    的文件路径.
    '''

    def build_alphabets(self, file_path, name='atis', save_dir='./save/alphabets/'):
        sentence_list, labels_list, intent_list = self.read_data(file_path)

        self.word_alphabet = Alphabet(name + '-word')
        self.label_alphabet = Alphabet(name + '-label')
        self.intent_alphabet = Alphabet(name + '-intent')

        for sentence in sentence_list:
            self.word_alphabet.add_words(sentence)
        for labels in labels_list:
            self.label_alphabet.add_words(labels)
        self.intent_alphabet.add_words(intent_list)

        if save_dir is not None:
            self.word_alphabet.save('./save/alphabets/')
            self.label_alphabet.save('./save/alphabets/')
            self.intent_alphabet.save('./save/alphabets/')

    '''
    构建训练集(测试集).
    type: ['train', 'test']
    '''

    def build_dataset(self, file_path, type_):
        assert type_ in ['train', 'test', 'dev']

        sentence_list, labels_list, intent_list = self.read_data(file_path)

        if type_ == "train":
            seed(self.random_state)
            index_list = list(range(0, len(sentence_list)))
            shuffle(index_list)

            new_sent, new_label, new_intent = [], [], []
            for idx in index_list:
                new_sent.append(sentence_list[idx])
                new_label.append(labels_list[idx])
                new_intent.append(intent_list[idx])

            sentence_list, labels_list, intent_list = new_sent, new_label, new_intent

        curr_dict = {'sentence_list': sentence_list,
                     'labels_list': labels_list,
                     'intent_list': intent_list}

        if type_ == 'train':
            self.train_set = curr_dict
            self.batch_start = 0
            self.train_len = len(sentence_list)
        elif type_ == 'test':
            self.test_set = curr_dict
            self.test_len = len(sentence_list)
        else:
            self.dev_set = curr_dict
            self.dev_len = len(sentence_list)

    '''
    抽象的构建, 给定总集, 训练集, 测试集快速构建一个对象.
    '''

    def quick_build(self, train_path='./data/atis.train.txt',
                    test_path='./data/atis.test.txt',
                    all_path='./data/atis.all.txt',
                    dev_path='./data/atis.dev.txt'):

        self.build_alphabets(all_path)
        self.build_dataset(train_path, 'train')
        self.build_dataset(test_path, 'test')
        self.build_dataset(dev_path, 'dev')

    '''
     对训练样例加 padding 0, 并且按序列长度排序排布.
    '''

    def add_padding(self, data_list, data_list_, data_list__):
        max_length = 0
        len_list = []
        for data in data_list:
            length = len(data)

            max_length = max(max_length, length)
            len_list.append(length)

        idx_list = argsort(len_list).tolist()[::-1]

        ret_sent, ret_slot, ret_len, ret_intent = [], [], [], []
        for idx in idx_list:
            ret_len.append(len_list[idx])
            ret_slot.append(data_list_[idx])
            ret_intent.append(data_list__[idx])
            ret_sent.append(data_list[idx])
            ret_sent[-1].extend([0] * (max_length - ret_len[-1]))

        return ret_sent, ret_slot, ret_len, ret_intent

    '''
     有随机性的返回训练集的一个 batch. digitalize=False,
     那么返回的 batch 中列表的元素不是数字，而是原始的字符串.
    '''

    def get_batch(self, batch_size=500, digitalize=True):
        batch_start = self.batch_start
        if batch_start + batch_size > self.train_len:
            batch_end = self.train_len
            self.batch_start = 0
        else:
            batch_end = self.batch_start + batch_size
            self.batch_start = batch_end

        sentence_batch = self.train_set['sentence_list'][batch_start: batch_end]
        labels_batch = self.train_set['labels_list'][batch_start: batch_end]
        intent_batch = self.train_set['intent_list'][batch_start: batch_end]

        if digitalize:
            sentence_batch = self.word_alphabet.indexs(sentence_batch)
            labels_batch = self.label_alphabet.indexs(labels_batch)
            intent_batch = self.intent_alphabet.indexs(intent_batch)

        sentence_batch, labels_batch, seq_lengths, intent_batch = self.add_padding(sentence_batch,
                                                                                   labels_batch, intent_batch)

        return sentence_batch, labels_batch, seq_lengths, intent_batch

    '''
    由于测试样例是用来检测泛化能力的, 因此一次性全部返回.
    '''

    def get_test(self, digitalize=True):
        sentence_list = self.test_set['sentence_list']
        labels_list = self.test_set['labels_list']
        intent_list = self.test_set['intent_list']

        if digitalize:
            sentence_list = self.word_alphabet.indexs(sentence_list)
            labels_list = self.label_alphabet.indexs(labels_list)
            intent_list = self.intent_alphabet.indexs(intent_list)

        sentence_list, labels_list, seq_lengths, intent_list = self.add_padding(sentence_list,
                                                                                labels_list, intent_list)

        return sentence_list, labels_list, seq_lengths, intent_list

    '''
    由于测试样例是用来检测泛化能力的, 因此一次性全部返回. 
    '''

    def get_dev(self, digitalize=True):
        sentence_list = self.dev_set['sentence_list']
        labels_list = self.dev_set['labels_list']
        intent_list = self.dev_set['intent_list']

        if digitalize:
            sentence_list = self.word_alphabet.indexs(sentence_list)
            labels_list = self.label_alphabet.indexs(labels_list)
            intent_list = self.intent_alphabet.indexs(intent_list)

        sentence_list, labels_list, seq_lengths, intent_list = self.add_padding(sentence_list,
                                                                                labels_list, intent_list)

        return sentence_list, labels_list, seq_lengths, intent_list
