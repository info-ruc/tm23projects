# 编码器
from transformers import BertTokenizer

token = BertTokenizer.from_pretrained("E:/localstorage/model/bert-base-chinese/")
# #print(token)
#
# out = token.batch_encode_plus(
# batch_text_or_text_pairs=['从明天起，做一个幸福的人。', '喂马，劈柴，周游世界。'],
# truncation=True,
# padding='max_length',
# max_length=17,
# return_tensors='pt',
# return_length=True)
# 查看编码输出
# for k, v in out.items():
#     print(k, v.shape)
# #把编码还原为句子
# print(token.decode(out['input_ids'][0]))
# 第7章/定义数据集
import torch
from datasets import load_from_disk


# dataset = load_dataset("lansinuote/ChnSentiCorp",cache_dir='E:\cache\dataset')
# dataset.save_to_disk('E:\localstorage\dataset')

class Dataset(torch.utils.data.Dataset):
    def __init__(self, split):
        self.dataset = load_from_disk('E:\localstorage\dataset\lansinuote_ChnSentiCorp')[split]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        text = self.dataset[i]['text']
        label = self.dataset[i]['label']
        return text, label


dataset = Dataset('train')
# print(len(dataset), dataset[1500])
print("数据集加载完成...")
device = 'cpu'
if torch.cuda.is_available():
    device = 'CUDA'
print("使用的计算平台为：" + device)


# 第7章/数据整理函数
def collate_fn(data):
    sents = [i[0] for i in data]
    labels = [i[1] for i in data]
    # 编码
    data = token.batch_encode_plus(batch_text_or_text_pairs=sents,
                                   truncation=True,
                                   padding='max_length',
                                   max_length=500,
                                   return_tensors='pt',
                                   return_length=True)
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


# 第7章/数据整理函数试算
# 模拟一批数据  测试用
# data = [
#     ('你站在桥上看风景', 1),
#     ('看风景的人在楼上看你', 0),
#     ('明月装饰了你的窗子', 1),
#     ('你装饰了别人的梦', 0),
# ]
# # 试算
# input_ids, attention_mask, token_type_ids, labels = collate_fn(data)
# print(input_ids.shape, attention_mask.shape, token_type_ids.shape, labels)
##### 测试用的代码结束
####使用真实数据集
print("dataset长度为：", len(dataset))

input_ids, attention_mask, token_type_ids, labels = collate_fn(dataset)
# 第7章/数据集加载器
loader = torch.utils.data.DataLoader(dataset=dataset,
                                     batch_size=16,
                                     collate_fn=collate_fn,
                                     shuffle=True,
                                     drop_last=True)
print("加载器的批次为：" + str(len(loader)))
# 第7章/加载预训练模型
from transformers import BertModel

pretrained = BertModel.from_pretrained("E:/localstorage/model/bert-base-chinese/")
# 统计参数量
print("参数量：" + str(sum(i.numel() for i in pretrained.parameters()) / 10000))

# 第7章/不训练预训练模型，不需要计算梯度
for param in pretrained.parameters():
    param.requires_grad_(False)
# 第7章/预训练模型试算
# 设定计算设备
pretrained.to(device)
# 模型试算
out = pretrained(
    input_ids=input_ids,
    attention_mask=attention_mask,
    token_type_ids=token_type_ids)


# print(out.last_hidden_state.shape)


# 第7章/定义下游任务模型
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # 使用预训练模型抽取数据特征
        with torch.no_grad():
            out = pretrained(input_ids=input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids)
            # 对抽取的特征只取第1个字的结果做分类即可
            out = self.fc(out.last_hidden_state[:, 0])
            out = out.softmax(dim=1)
        return out


model = Model()
# 设定计算设备
model.to(device)
# # 试算
# print(model(input_ids=input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids).shape)
# 第7章/训练
from transformers import AdamW
from transformers.optimization import get_scheduler


def train():
    # 定义优化器
    optimizer = AdamW(model.parameters(), ir=5e-4)
    # 定义loss函数
    criterion = torch.nn.CrossEntropyLoss()
    # 定义学习率调节器
    scheduler = get_scheduler(name='linear',
                              num_warmup_steps=0,
                              num_training_steps=len(loader),
                              optimizer=optimizer)
    # 将模型切换到训练模式
    model.train()
    # 按批次遍历训练集中的数据
    for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(loader):
        # 模型计算
        out = model(input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids)
        # 计算 loss 并使用梯度下降法优化模型参数
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        # 输出各项数据的情况，便于观察
        if i % 10 == 0:
            out = out.argmax(dim=1)
            accuracy = (out == labels).sum().item() / len(labels)
            ir = optimizer.state_dict()['param groups'][0]['lr']
            print(i, loss.item(), ir, accuracy)


train()
