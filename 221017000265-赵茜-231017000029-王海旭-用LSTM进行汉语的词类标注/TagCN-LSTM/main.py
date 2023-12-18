#coding: UTF-8
import os
import argparse
import torch
from lstm import LSTM_Model
from lstm_crf import BiLSTM_CRF_model

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu', default = [], nargs='+', type=str, help='Specify GPU id.')
parser.add_argument( '-e', '--epoch', type=int, help='train epoch number')
parser.add_argument( '--checkpoint', default = './', type=str, help='checkpoint.')
parser.add_argument( '--seed', default = 1, type=int, help='seed for pytorch init')
parser.add_argument( '-b','--bidirection', action='store_true', help='use bi-direction lstm or not')
parser.add_argument('--weighted_tag', action='store_true',help='use wighted loss or not' )
parser.add_argument('--crf', action='store_true',help='use crf or not' )

args = parser.parse_args()
print(args)

torch.manual_seed(args.seed)

if args.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu[0]

if args.crf: #是否使用CRF层
	model= BiLSTM_CRF_model(args)
	model.train()
else:
	model = LSTM_Model(args)
	model.train()
