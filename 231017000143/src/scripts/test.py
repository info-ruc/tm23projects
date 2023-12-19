#!/usr/bin/env python3
from MyDataset import MyDataset
from MyNet import MyNet
import torch
import torch.nn.functional as F
import torch.optim as optim

def makewords(seqswords):
    seqswords = seqswords.split(exstr)
    hidden = None

    def wordItem(input_word):
        nonlocal hidden

        input_word_index = dataset.word2index[input_word]
        input_ = torch.Tensor([[input_word_index]]).long().to(device)
        output, hidden = model(input_, hidden)
        top_word_index = output[0].topk(1).indices.item()
        return dataset.index2word[top_word_index]

    result = [] 
    cur_word = exstr

    for i in range(seqs):
        if cur_word == exstr:
            result.append(cur_word)
            wordItem(cur_word)
            if len(seqswords) == 0:
                break
            for w in seqswords:
                result.append(w)
                cur_word = wordItem(w)
        else:
            result.append(cur_word)
            cur_word = wordItem(cur_word)
    ress = "".join(result)
    ress = ress.strip(exstr)
    return ress

def testmains(keywords):
    res = []
    keyword_arr = keywords.split("|")
    for d in keyword_arr:
        try:
            w = makewords(d)
        except:
            w = ""
        res.append(w)
    return "\n".join(res)

def load_model(file):
    pt = torch.load(file)
    return pt

if __name__ == "__main__":

    seqs = 7
    exstr = "#"

    save = "./model.pkl"

    print("-- Model loading --")

    model = load_model(save)

    print("-- Model loading completed --")

    device = torch.device("cuda:0")

    # 数据集
    dataset = MyDataset(seqs=seqs)

    model = model.to(device)

    while True:
        print("提示：多行使用|分割   例：[孤|柳|鸟|石] [床|纸|袜|面]  [输入q退出]")
        keywords = input("请输入关键字：")
        if keywords == "q":
            break
        result = testmains(keywords)
        print("\n-------------------")
        print(result)
        print("-------------------\n\n")
