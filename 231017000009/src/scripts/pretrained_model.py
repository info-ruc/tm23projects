#!/usr/bin/env python
# coding: utf-8

# In[1]:


#加载tokenizer
from transformers import AutoTokenizer
pretrained_model_name_or_path = r'C:\Users\peixi\Downloads\Huggingface\model\hflchinese-roberta-wwm-ext'
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)

import torch
from datasets import load_from_disk

class Dataset(torch.utils.data.Dataset):
    def __init__(self, split, max_samples=None):
        self.dataset = load_from_disk(r'C:\Users\peixi\Downloads\Huggingface\Data\ChnSentiCorp')[split]
        if max_samples:
            self.dataset = self.dataset.select(list(range(min(max_samples, len(self.dataset)))))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        review = self.dataset[i]['text']
        label = self.dataset[i]['label']
        return review, label

dataset = Dataset('train', 9600)
len(dataset), dataset[20]

#定义计算设备
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

# 数据整理函数
def collate_fn(data):
    sents = [i[0] for i in data]
    labels = [i[1] for i in data]
    # 编码
    data = tokenizer.batch_encode_plus(batch_text_or_text_pairs=sents, truncation=True, padding='max_length', max_length=500, return_tensors='pt', return_length=True)
    # input_ids：编码之后的数字
    # attention_mask：补零的位置是0, 其他位置是1
    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    token_type_ids = data['token_type_ids']
    labels = torch.LongTensor(labels)
    # 把数据移动到计算设备上
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    token_type_ids = token_type_ids.to(device)
    labels = labels.to(device)
    return input_ids, attention_mask, token_type_ids, labels



# In[2]:


#数据加载器
loader = torch.utils.data.DataLoader(dataset=dataset,
                                     batch_size=16,
                                     collate_fn=collate_fn,
                                     shuffle=True,
                                     drop_last=True)

len(loader)


# In[3]:


#加载模型
from transformers import BertModel
pretrained_model_name_or_path = r'C:\Users\peixi\Downloads\Huggingface\model\hflchinese-roberta-wwm-ext'
pretrained = BertModel.from_pretrained(pretrained_model_name_or_path)


# In[4]:


#设定计算设备
pretrained.to(device)


# In[5]:


# 预训练模型试算
for input_ids, attention_mask, token_type_ids, labels in loader:
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    token_type_ids = token_type_ids.to(device)
    labels = labels.to(device)

    out = pretrained(input_ids=input_ids,
                     attention_mask=attention_mask,
                     token_type_ids=token_type_ids)

    print(out.last_hidden_state.shape)
    break  # 退出循环


# In[6]:


#定义下游任务模型
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(in_features=768, out_features=2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        #使用预训练模型抽取数据特征
        with torch.no_grad():
            out = pretrained(input_ids=input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids)

        #对抽取的特征只取第一个字的结果做分类即可
        out = self.fc(out.last_hidden_state[:, 0])

        out = out.softmax(dim=1)

        return out


model = Model()

#设定计算设备
model.to(device)

#试算
model(input_ids=input_ids,
      attention_mask=attention_mask,
      token_type_ids=token_type_ids).shape


# In[7]:


#训练
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

def train():
    # 定义优化器
    optimizer = optim.AdamW(model.parameters(), lr=5e-4)

    # 定义loss函数
    criterion = torch.nn.CrossEntropyLoss()

    # 定义学习率调节器
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: 0.98 ** step)  # 使用LambdaLR作为学习率调节器

    # 模型切换到训练模式
    model.train()

    # 按批次遍历训练集中的数据
    for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(loader):

        # 模型计算
        out = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        # 计算loss并使用梯度下降法优化模型参数
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # 输出各项数据的情况，便于观察
        if i % 10 == 0:
            out = out.argmax(dim=1)
            accuracy = (out == labels).sum().item() / len(labels)
            lr = optimizer.param_groups[0]['lr']
            print(i, loss.item(), lr, accuracy)

train()


# In[8]:


#测试
def test():
    #定义测试数据集加载器
    loader_test = torch.utils.data.DataLoader(dataset=Dataset('test'),
                                              batch_size=16,
                                              collate_fn=collate_fn,
                                              shuffle=True,
                                              drop_last=True)

    #下游任务模型切换到运行模式
    model.eval()
    correct = 0
    total = 0
    sentence_count = 0  # 用于记录句子数量

    #增加输出前5句的结果并与真实的label进行比较
    for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(loader_test):

        #计算
        with torch.no_grad():
            out = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids)

        #统计正确率
        out = out.argmax(dim=1)
        correct += (out == labels).sum().item()
        total += len(labels)

        #输出前5句的结果并与真实的label进行比较
        for j in range(len(input_ids)):  # 遍历当前batch的每个句子
            # Decode input_ids
            decoded_input_ids = tokenizer.decode(input_ids[j], skip_special_tokens=True)
            print("Review:", decoded_input_ids)  # 输出 input_ids 的 decode 结果
            print("Prediction:", out[j].item())  # 输出预测结果
            print("True label:", labels[j].item())  # 输出真实标签
            sentence_count += 1  # 句子数量加1
            if sentence_count == 5:  # 当输出了5条句子后，退出循环
                break

        if sentence_count == 5:  # 当输出了5条句子后，退出外层循环
            break

    print("Accuracy for the first 5 sentences:", correct / total)

test()


# In[9]:


#手动保存模型
pretrained.save_pretrained(r'C:\Users\peixi\Downloads\Huggingface\model\hflchinese-roberta-wwm-ext\trained')

