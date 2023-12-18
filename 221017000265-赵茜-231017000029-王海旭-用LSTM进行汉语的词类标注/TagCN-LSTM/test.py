#coding: UTF-8
import os
import argparse
import torch
from dataset import Dataset, config
from lstm import LSTMTagger
import numpy as np
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu', default = [], nargs='+', type=str, help='Specify GPU id.')
parser.add_argument( '-e', '--epoch', default = 1, type=int, help='train epoch number')
parser.add_argument( '--checkpoint', default = 'checkpoint/', type=str, help='checkpoint.')
parser.add_argument( '--seed', default = 1, type=int, help='seed for pytorch init')
parser.add_argument( '-b','--bidirection', action='store_true', help='use bi-direction lstm or not')
parser.add_argument('--weighted_tag', action='store_true',help='use wighted loss or not' )
parser.add_argument('--crf', action='store_true',help='use crf or not' )
args = parser.parse_args()

if args.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu[0]

dataset_tool = Dataset(config)


#print("LSTM model")
model = LSTMTagger(config.EMBEDDING_DIM, config.HIDDEN_DIM, dataset_tool.word_to_ix_length,\
                          len(config.TAG_to_ix), config.LAYER, config.DROP_RATE, 1, args.bidirection)

if os.path.exists(args.checkpoint+"/epoch_max_accuracy.pkl"):
    model.load_state_dict(torch.load(args.checkpoint+"/epoch_max_accuracy.pkl"))
    print("加载模型...\n")


word_dict = {'n':'名词','t':'时间词','s':'处所词','f':'方位词','m':'数词','q':'量词','b':'区别词','r':'代词','v':'动词','a':'形容词','z':'状态词','d':'副词','p':'介词','c':'连词','u':'助词','y':'语气词','e':'叹词','o':'拟声词','i':'成语','l':'习用语','j':'简称','h':'前接成分','k':'后接成分','g':'语素','x':'非语素字','w':'标点符号'}

if __name__ == '__main__':
    sentence = input("请输入句子，词语间用空格隔开，如“我 在 学习 自然 语言 处理”:")
    word_str = sentence.split()
    print("输入:",word_str)
    print("\n计算中...")
    inputs = dataset_tool.prepare_sequence(word_str,dataset_tool.word_to_ix)
    inputs = torch.tensor(inputs, dtype=torch.long)
    inputs = inputs.cuda(0)
    model = model.cuda(0)
    if args.crf:
        with torch.no_grad():
            score, pred_tag = model(inputs)
        result = []
        for item in pred_tag:
            result.append(list(config.TAG_to_ix.keys())[item])
    else:
        with torch.no_grad():
            result = []
            score = model(inputs,[],is_test=True)
            for word in score.cpu().numpy():
                pred = np.where(word == np.max(word))
                result.append(list(config.TAG_to_ix.keys())[pred[0][0]])
    print("\n标注结果:")
    print(result)
    for r in result:
        print(word_dict[r],end=" ")
    print()
