from collections import Counter
import jieba
import json
import torch


#  load data
def load_data(train=True, category='car_brand', participle=True, key_words=True):
    dataset = []
    length = 0
    if train:
        file_path = 'dataset/datatrain.json'
    else:
        file_path = 'dataset/simulation.json'
    for line in open(file_path, "r", encoding='utf-8'):
        instance = json.loads(line)
        sent = []
        if key_words and len(instance['keywords']):
            sent += instance['keywords'].split(',')
        if participle:
            sent += [x for x in jieba.cut(instance['sentence'])] 
        else:
            sent += [x for x in instance['sentence']] 
        length = max(length, len(sent))
        label = instance[category]
        dataset.append([sent, label])
    return dataset


# collect data
def make_data(dataset, drop_num=1):
    counter = Counter()
    for (sent, label) in dataset:
        counter.update(sent)
    vocab = [(word, count) for (word, count) in counter.items() if count > drop_num]
    sorted_vocab = sorted(vocab, key=lambda x: x[1], reverse=False)
    word2idx = {'unk': 0}
    word2idx.update({word: i + 1 for i, (word, count) in enumerate(sorted_vocab)})
    return word2idx


# serialization
def serialization(dataset, word2idx):
    inputs = []
    labels = []
    for sent, label in dataset:
        idx_seq = []
        for word in sent:
            if word in word2idx:
                idx_seq.append(word2idx[word])
            else:
                idx_seq.append(word2idx["unk"])
        inputs.append(idx_seq)
        labels.append(label)
    return inputs, labels


# padding_and_cut
def padding_and_cut(inputs, max_length):
    # padding 
    for i in range(len(inputs)):
        if len(inputs[i]) < max_length:
            inputs[i] += [0]*(max_length-len(inputs[i]))
        else:
            # capture
            inputs[i] =inputs[i][:max_length]
    return inputs


# read data
def read_data(train=True, sequence_length=20, category='car_brand', drop_num=1, participle=True, key_words=True):
    data_train = load_data(category=category, participle=participle, key_words=key_words)
    data_test = load_data(train=False, category=category, participle=participle, key_words=key_words)
    data = data_train + data_test
    data_w2i = make_data(data, drop_num=drop_num)
    num_vocab = len(data_w2i)
    if train:
        inputs, labels = serialization(data_train, data_w2i)
    else:
        inputs, labels = serialization(data_test, data_w2i)
    inputs = padding_and_cut(inputs, max_length=sequence_length)
    return inputs, labels, num_vocab


# label
def label_digitize(labels):
    counter = Counter()
    counter.update(labels)
    label2idx = {label:i for i, (label, _) in enumerate(counter.items())}
    for i, label in enumerate(labels):
        labels[i] = label2idx[label]
    return labels, label2idx


if __name__ == '__main__':
    dtype = torch.FloatTensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    inputs, labels, _ = read_data(train=False, sequence_length=50)
    labels = label_digitize(labels)
