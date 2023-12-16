import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle as pkl

class Global_Config(object):

    """配置参数"""
    def __init__(self, model_name, train=True):
        self.model_name = model_name 
        
        self.train_path = './THUCNews/data/train2.txt'                                # 训练集
        self.dev_path = './THUCNews/data/dev2.txt'                                    # 验证集
        self.test_path = './THUCNews/data/test2.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            './THUCNews/data/class.txt', encoding='utf-8').readlines()] if train else []           # 类别名单
        self.save_path = './THUCNews/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.log_path = './THUCNews/log/' + self.model_name
        self.vocab_path = './temp/vocab_to_id200d_10m.dict'
        self.emb_path = './temp/embeding_200d_10m.model'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        self.dropout = 0.5                                              # 随机失活
        self.require_improvement = 2000                             # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs = 20                                            # epoch数
        self.batch_size = 64                                           # mini-batch大小
        self.pad_size = 500                                             # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3                                       # 学习率

        

