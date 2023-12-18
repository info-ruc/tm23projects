from flask import Flask, g, redirect, url_for, render_template, session, request
from gensim.models.word2vec import Word2Vec

import numpy as np
import torch
import torch.nn as nn

import os
import pickle as pkl
import tensorflow as tf
import re
import sys
sys.path.append('../src/models')
sys.path.append('../src')

from global_config import Global_Config
import TextRCNN

import jieba
app = Flask(__name__)


@app.route('/')
def index():
    #return "hello"
    return render_template('index.html')

@app.route('/text_predict', methods=['POST'])
def text_predict():
    classes = ['财经', '彩票', '房产', '股票', '家居', '教育', '科技', '社会', '时尚', '时政', '体育', '星座', '游戏', '娱乐']
    UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号
    def word2id_Tensor(input, config):
        if os.path.exists(config.vocab_path):
            vocab = pkl.load(open(config.vocab_path, 'rb'))
        else:
            print("Can not file the vocab file! ")
        print(f"Vocab size: {len(vocab)}")

        seq_len = len(input)
        if config.pad_size:
            if seq_len < config.pad_size:
                input.extend([PAD] * (config.pad_size - seq_len))
            else:
                input = input[:config.pad_size]
                seq_len = config.pad_size
        words_line = []

        # word to id
        # 未出现词用UNK代替
        for word in input:
            words_line.append(vocab.get(word, vocab.get(UNK)))
        x = torch.LongTensor([[_ for _ in words_line]]).to(config.device)
        return x, [seq_len]
        
  
    def predict(text):
        with torch.no_grad():
            output = model(text)
            print(output)
            predic = torch.argmax(output.data).item() + 0
            print(predic)
            #y = int(np.argmax(model.predict(test), axis=1))
        return classes[predic]
        
    if request.method == 'GET':
        pass
    else:
        data = request.json  # 获取 JOSN 数据
        data = data.get('content')
        data = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。:：？、~@#￥%……&*（）“”]+", "", data).replace(' ','').replace("\n", "")
        seg_list = jieba.cut(data, cut_all=False)   # 精确模式
        seg_list = list(seg_list)
        print(seg_list)
        #seg_string = [' '.join(seg_list)]
        seg_list = word2id_Tensor(seg_list, config)
        print(seg_list)
        label = predict(seg_list)
        print(label)
        return "类别："+label

if __name__ == '__main__':
    g_config = Global_Config("TextRCNN",train=False)
    g_config.class_list = [x.strip() for x in open(
            '../src/THUCNews/data/class.txt', encoding='utf-8').readlines()]              # 类别名单
    g_config.num_classes = len(g_config.class_list)
    g_config.save_path = '../src/THUCNews/saved_dict/TextRCNN2.ckpt'        # 模型训练结果
  
    g_config.vocab_path = '../src/temp/vocab_to_id200d_10m.dict'
    g_config.emb_path = '../src/temp/embeding_200d_10m.model'
    
    
    
    config = TextRCNN.Config(g_config)

    model = TextRCNN.Model(config).to(config.device)
    model.load_state_dict(torch.load(config.save_path))
    model.eval()

    app.run(host="127.0.0.1", port=5000, debug=False, threaded=False)

