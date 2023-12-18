import jieba
import sys
import yaml
import multiprocessing
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout, Activation
from keras.models import model_from_yaml

np.random.seed(666)
# 设置参数
vocab_dim = 100
maxlen = 100
n_iterations = 1
min_frequence = 10
window_size = 5
batch_size = 32
n_epoch = 1
input_length = 100
cpu_count = multiprocessing.cpu_count()


# 加载训练文件
def loadfile():
    neg = pd.read_excel('data/neg.xls', header=None, index=None)
    pos = pd.read_excel('data/pos.xls', header=None, index=None)
    x = np.concatenate((pos[0], neg[0]))
    # 1表示好评，0表示差评
    y = np.concatenate((np.ones(len(pos), dtype=int), np.zeros(len(neg), dtype=int)))
    return x, y


# 对句子进行分词，并去掉换行符
def tokenizer(text):
    text = [jieba.lcut(sentence.replace('\n', '')) for sentence in text]
    return text


# 创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
# 在函数中定义函数，用到了闭包
def create_dict(model=None, x=None):
    # 1- Creates a word to index mapping
    # 2- Creates a word to vector mapping
    # 3- Transforms the Training and Testing Dictionaries
    if (x is not None) and (model is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.wv.vocab.keys(), allow_update=True)
        word2index = {v: k + 1 for k, v in gensim_dict.items()}  # 所有频数超过10的词语的索引
        word2v = {word: model[word] for word in word2index.keys()}  # 所有频数超过10的词语的词向量

        def parse_dataset(x):
            # Words become integers
            data = []
            for sentence in x:
                new_sentence = []
                for word in sentence:
                    try:
                        # 将句子中的每个单词转换成对应的索引
                        new_sentence.append(word2index[word])
                    except:
                        # 句子中含有频数小于10的词语，索引为0
                        new_sentence.append(0)
                data.append(new_sentence)
            return data

        x = parse_dataset(x)
        x = sequence.pad_sequences(x, maxlen=maxlen)
        return word2index, word2v, x
    else:
        print('No data provided...')


# 训练词向量模型，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
def word2vec_train(x):
    model = Word2Vec(
        size=vocab_dim,
        min_count=min_frequence,
        window=window_size,
        workers=cpu_count,
        iter=n_iterations
    )
    model.build_vocab(x)
    model.train(x, total_examples=model.corpus_count, epochs=model.iter)
    model.save('lstm_model/Word2vec_model.pkl')
    index_dict, word_vectors, x = create_dict(model=model, x=x)
    return index_dict, word_vectors, x


def get_data(index_dict, word_vectors, x, y):
    n_symbols = len(index_dict) + 1  # 所有单词的索引数，频数小于10的词语索引为0，所以加1
    embedding_weights = np.zeros((n_symbols, vocab_dim))  # 索引为0的词语，词向量全为0
    for word, index in index_dict.items():  # 从索引为1的词语开始，对每个词语对应其词向量
        embedding_weights[index, :] = word_vectors[word]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    print(x_train.shape, y_train.shape)
    return n_symbols, embedding_weights, x_train, y_train, x_test, y_test


##定义网络结构
def train_lstm(n_symbols, embedding_weights, x_train, y_train, x_test, y_test):
    print('Defining a Simple Keras Model...')
    model = Sequential()  # or Graph or whatever
    model.add(Embedding(output_dim=vocab_dim,
                        input_dim=n_symbols,
                        mask_zero=True,
                        weights=[embedding_weights],
                        input_length=input_length))  # Adding Input Length
    model.add(LSTM(output_dim=50, activation='sigmoid', inner_activation='hard_sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    print('Compiling the Model...')
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    print("Train...")
    model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=n_epoch, verbose=1, validation_data=(x_test, y_test))

    print("Evaluate...")
    score = model.evaluate(x_test, y_test, batch_size=batch_size)

    yaml_string = model.to_yaml()
    with open('lstm_model/lstm.yml', 'w') as outfile:
        outfile.write(yaml.dump(yaml_string, default_flow_style=True))
    model.save_weights('lstm_model/lstm.h5')
    print('Test score:', score)


# 训练模型，并保存
def train():
    print('Loading Data...')
    x, y = loadfile()
    print(len(x), len(y))
    print('Tokenising...')
    x = tokenizer(x)
    print('Training a Word2vec model...')
    index_dict, word_vectors, x = word2vec_train(x)
    print('Setting up Arrays for Keras Embedding Layer...')
    n_symbols, embedding_weights, x_train, y_train, x_test, y_test = get_data(index_dict, word_vectors, x, y)
    print(x_train.shape, y_train.shape)
    train_lstm(n_symbols, embedding_weights, x_train, y_train, x_test, y_test)


def input_transform(string):
    words = jieba.lcut(string)
    words = np.array(words).reshape(1, -1)
    model = Word2Vec.load('lstm_model/Word2vec_model.pkl')
    _, _, x = create_dict(model, words)
    return x


def lstm_predict(string):
    with open('lstm_model/lstm.yml', 'r') as f:
        yaml_string = yaml.load(f, Loader=yaml.FullLoader)
    model = model_from_yaml(yaml_string)
    model.load_weights('lstm_model/lstm.h5')
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    data = input_transform(string)
    data.reshape(1, -1)
    result = model.predict_classes(data)
    if result[0][0] == 1:
        print(string, '\n AI分析结果： 好评')
    else:
        print(string, '\n AI分析结果： 差评')


if __name__ == '__main__':
    train()
    string = '牛逼的手机，从3米高的地方摔下去都没坏，质量非常好'
    lstm_predict(string)