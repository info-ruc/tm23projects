import argparse

import data

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM/GRU/Transformer Language Model')
parser.add_argument('--data', type=str, default='./dataset',
                    help='location of the data corpus')
args = parser.parse_args()

corpus = data.Corpus(args.data)
print(type(corpus), corpus)

print(type(corpus.train), corpus.train)
print(type(corpus.valid), corpus.valid)
print(type(corpus.test), corpus.test)

print(type(corpus.dictionary.word2idx), corpus.dictionary.word2idx)
print(type(corpus.dictionary.idx2word), corpus.dictionary.idx2word, len(corpus.dictionary.idx2word))
