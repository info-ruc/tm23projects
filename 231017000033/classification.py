# -*- coding:utf-8 -*-

import sys
import time
import math
import random
import pandas as pd

import torch
from torch import nn
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader,Dataset
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.data.functional import to_map_style_dataset
from utils.transformer_net import TransformerNet

#【项目说明】
# 本程序是基于pytorch库实现新闻分类，使用的是pytorch官方提供的英文新闻数据集。
# 使用了如下两种模型：
# 1 词嵌入层+线性神经网络层
# 2 transformer的encoder+线性神经网络层
# 程序中默认是使用第一个模型，通过修改一行注释即可切换模型。
# 运行该程序的说明在代码的最下方

# 全局参数
EPOCHS = 10     # 重复训练10次
BATCH_SIZE = 64 # 每个epoch批处理的数据量
embed_dim = 64  # 词嵌入向量维度
num_heads = 8   # TransformerNet需要的参数
learn_rate = 5  # 学习率
params_save = True # 是否保存模型参数到文件
params_path = "./parameters/classification_param.pt" # 模型参数文件路径

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 实时下载并加载pytorch官方的数据集
train_iter, test_iter = AG_NEWS()
# 计算类别数
num_class = len(set([label for (label, text) in train_iter]))

# 构建词汇表和文本标签管道
def build_vocabulary():
    # 根据词汇表将文字转为编码，如将“文本挖掘”转为[2,10,3,11]
    tokenizer = get_tokenizer("basic_english")
    def yield_tokens(data_iter):
        for _, text in data_iter:
            yield tokenizer(text)
    # 构建词汇表，'<unk>'代表词汇字典中未存在的token
    vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    # prepare the text processing pipeline with the tokenizer and vocabulary. 
    text_pipeline = lambda x: vocab(tokenizer(x))
    label_pipeline = lambda x: int(x) - 1
    return vocab, text_pipeline, label_pipeline

# 构建词汇表
vocab, text_pipeline, label_pipeline = build_vocabulary()
# 词汇表大小
vocab_size = len(vocab)

# 以ratio的比例随机划分数据集
def random_split_dataset(data, ratio: float):
    original_num = len(data)
    remain_num = int(original_num * ratio)
    return random_split(data, [remain_num, original_num - remain_num])

# 管理所有数据集，使得更方便地迭代数据集
class DataManager():
    def __init__(self):
        # 将迭代式的数据集转换为映射式的数据集, 使得可以通过索引更方便地访问数据集中的元素
        dataset_test = to_map_style_dataset(test_iter)
        dataset_train = to_map_style_dataset(train_iter)
        # 随机取出训练集中占比0.05的数据作为验证数据集，占比0.95的数据作为训练集
        dataset_train_, dataset_valid_= random_split_dataset(dataset_train, 0.95)
        # 创建DataLoader实例，DataLoader用于管理数据集，其支持随机打乱数据集顺序和批处理数据集
        self.dl_train_ = DataLoader(dataset_train_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=self.collate_batch)
        self.dl_test_ = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True, collate_fn=self.collate_batch)
        self.dl_valid_ = DataLoader(dataset_valid_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=self.collate_batch)

    # 数据集分批函数
    def collate_batch(self, batch):
        label_list, text_list, offsets = [], [], [0]
        for _label, _text in batch:
            label_list.append(label_pipeline(_label))
            # 将文本转成int类型的向量，向量中的值是词在词汇表中的索引值
            processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
            text_list.append(processed_text)
            offsets.append(processed_text.size(0))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        # 偏移量，即文本包含词的个数
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text_list = torch.cat(text_list)
        return label_list.to(device), text_list.to(device), offsets.to(device)

class NormalNet(nn.Module):
    """
    embed_dim：词嵌入向量维度
    vocab_size：词汇表大小
    num_class：分类数
    """
    def __init__(self, vocab_size, embed_dim, num_class):
        super(NormalNet, self).__init__()
        # 第一层为词嵌入层
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()
    
    def init_weights(self):
        initrange = 0.5 # 以[-0.5, 0.5]为值范围，随机初始化权重
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

# 创建模型实例
# model = NormalNet(vocab_size, embed_dim, num_class).to(device)
# 使用transformer模型需要跑很长时间
model = TransformerNet(vocab_size, embed_dim, num_class, num_heads, 12).to(device)

# 打印模型的结构
print("-" * 80) # 分割线
print('model structure:')
print('num_class', num_class)
print('vocab_size', vocab_size)
print(model)
print("-" * 80)

# 交叉熵损失函数，CrossEntropyLoss()
# 信息熵衡量事物不确定性，熵越大，事物不确定性越大，事物越复杂
# 交叉熵能够衡量同一随机变量中的两个不同概率分布的差异程度，在机器学习中其表示为真实概率分布与预测概率分布之间的差异
# 交叉熵的值越小，模型预测效果就越好
# 交叉熵经常搭配softmax使用，将输出的结果进行处理，使其多个分类的预测值和为1，再通过交叉熵来计算损失
criterion = torch.nn.CrossEntropyLoss()
# SGD即随机梯度下降, 一般默认指批量梯度下降法, 是最经典的神经网络优化方法，虽收敛速度慢，但是收敛效果较稳定
optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)

# 训练函数
# [梯度裁剪知识]
# clip_grad_norm_函数用于在训练神经网络时限制梯度的大小, 被称为梯度裁剪(gradient clipping)。
# 梯度裁剪可以防止梯度爆炸，梯度爆炸通常发生在训练深度神经网络时, 尤其是在处理长序列数据的
# 循环神经网络（RNN）中。当梯度爆炸发生时，参数更新可能会变得非常大，导致模型无法收敛或出现
# 数值不稳定。梯度裁剪使模型训练变得更加稳定。
# model.parameters()表示模型的所有参数。
# 对于一个神经网络，参数通常包括权重和偏置项。0.1是一个指定的阈值，表示梯度的最大范数(L2范数)。
# 如果计算出的梯度范数超过这个阈值，梯度会被缩放，则使其范数等于阈值。
def train(dataloader, epoch):
    # 调用train()函数启用dropout和batch normalization层
    model.train()
    # 每隔N条数据打印一次日志
    log_interval = 593
    total_acc, total_count = 0, 0
    start_time = time.time()
    # 循环一次处理一批数据，offsets是指文本词量
    for idx, (label, text, offsets) in enumerate(dataloader):
        optimizer.zero_grad() # 梯度归零
        predicted_label = model(text, offsets) # 计算模型输出
        loss = criterion(predicted_label, label) # 计算模型损失
        # train_loss_rec.append(loss.item()) # 损失加入到列表中
        loss.backward() # 反向传播
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1) # 梯度裁剪
        # 使用优化器计算梯度然后更新模型的权重和偏差
        optimizer.step()
        # learn_rates_rec.append(optimizer.param_groups[0]["lr"])
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed_time = time.time() - start_time # 计算消耗时间
            print("| epoch {:3d} | batch {:5d}/{:5d}  ""| accuracy {:8.3f} cost time{:8.3f}"
                .format(epoch, idx, len(dataloader), total_acc / total_count, elapsed_time))
            total_acc, total_count = 0, 0
            start_time = time.time()

def evaluate(dataloader):
    # 在评估模型性能时, 调用eval()函数禁用dropout和batch normalization的函数, 使得不会更新模型的权重和偏差
    model.eval()
    total_acc, total_count = 0, 0
    # 调用no_grad()函数禁止梯度计算, 避免在评估期间浪费计算资源
    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            predicted_label = model(text, offsets)
            loss = criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc / total_count

def predict(text, text_pipeline):
    with torch.no_grad():
        text = torch.tensor(text_pipeline(text))
        output = model(text, torch.tensor([0]))
        return output.argmax(1).item() + 1

# 执行训练过程
def run_train():
    total_accu = None
    data_manager = DataManager()
    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()
        train(data_manager.dl_train_, epoch)
        accu_val = evaluate(data_manager.dl_valid_)
        if total_accu is not None and total_accu > accu_val:
            scheduler.step()
        else:
            total_accu = accu_val
        print("| epoch {:3d} | cost time: {:5.2f}s | valid accuracy {:8.3f} "
            .format(epoch, (time.time() - epoch_start_time), accu_val))
        print("-" * 80)
    print("checking the results of test dataset.")
    accu_test = evaluate(data_manager.dl_test_) # 使用测试数据集评估精度
    print("test accuracy {:8.3f}".format(accu_test))
    if params_save:
        torch.save(model.state_dict(), params_path) # 保存模型参数

# 执行文本预测
def run_predict():
    # 新闻分类标签
    ag_news_label = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tec"}
    if params_save:
        model.load_state_dict(torch.load(params_path)) # 加载模型参数
    # 从CSV文件中读取待测试的文本，文件中是我们随机找的多条新闻文本
    predict_texts = pd.read_csv("./docs/predict_text.csv", delimiter='|')
    txt_rows = predict_texts.shape[0]
    txt_cols = predict_texts.shape[1]
    print("predict text list:\ rows {} cols{}".format(txt_rows, txt_cols))
    for row in range(txt_rows): 
        target_txt = predict_texts.iloc[row, 2]
        print("-" * 80)
        print('predict text:')
        print(target_txt)
        print("-" * 80)
        print("This is a %s news" % ag_news_label[predict(target_txt, text_pipeline)])

# 运行该程序有三个命令
# python3 classification.py
# 此时args[0]为空，则会先执行训练再执行预测，不会保存模型参数，因为模型训练后直接用于预测了
# python3 classification.py train
# 只执行训练过程，训练结束会保存模型参数到文件
# python3 classification.py
# 只执行预测过程，加载预先保存好了的参数模型文件

if __name__ == "__main__":
    # 截取classification.py后面的所有参数
    args = sys.argv[1:]
    if len(args) < 1:
        params_save = False
        run_train()
        model = model.to("cpu")
        run_predict()
    elif len(args) > 0:
        if args[0] == "train" :
            run_train()
        elif args[0] == "predict" :
            run_predict()
        else:
            print("unknown args")
    else:
        print("unknown args")