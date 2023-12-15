from pathlib import Path
from transformers import BertTokenizer
from transformers import BertModel
from transformers.optimization import get_scheduler, AdamW
import torch
from datasets import load_from_disk
from modelscope import snapshot_download  # 模型下载


def load_encode_tool(pretrained_model_name_or_path):
    # 加载编码工具bert-base-chinese
    token = BertTokenizer.from_pretrained(Path(f'{pretrained_model_name_or_path}'))
    return token


# 定义数据集
class Dataset(torch.utils.data.Dataset):
    def __init__(self, split):
        mode_name_or_path = r'file://../dataset'
        self.dataset = load_from_disk(mode_name_or_path)[split]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        text = self.dataset[i]['text']
        label = self.dataset[i]['label']
        return text, label


# 数据整理函数
def collate_fn(data):
    sents = [i[0] for i in data]
    labels = [i[1] for i in data]
    # 编码
    data = token.batch_encode_plus(batch_text_or_text_pairs=sents, truncation=True, padding='max_length',
                                   max_length=500, return_tensors='pt', return_length=True)
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


# 定义下游任务模型
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # 使用预训练模型抽取数据特征
        with torch.no_grad():
            out = pretrained(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # 对抽取的特征只取第1个字的结果做分类
        out = self.fc(out.last_hidden_state[:, 0])
        out = out.softmax(dim=1)
        return out


# 训练
def train():
    # 定义优化器
    optimizer = AdamW(model.parameters(), lr=5e-4)
    # 损失函数
    criterion = torch.nn.CrossEntropyLoss()
    # 定义学习率调节器
    scheduler = get_scheduler(name='linear', num_warmup_steps=0, num_training_steps=len(loader), optimizer=optimizer)
    # 将模型切换到训练模式
    model.train()
    # 按批次遍历训练集中的数据
    for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(loader):
        # 模型计算
        out = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # 计算loss并使用梯度下降法优化模型参数
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()  # 梯度下降法优化模型参数
        scheduler.step()  # 学习率调节器
        optimizer.zero_grad()  # 清空梯度
        # 输出各项数据的情况，便于观察
        if i % 10 == 0:
            out = out.argmax(dim=1)
            accuracy = (out == labels).sum().item() / len(labels)
            lr = optimizer.state_dict()['param_groups'][0]['lr']
            print(i, loss.item(), lr, accuracy)


# 测试
def test():
    # 定义测试数据集加载器
    loader_test = torch.utils.data.DataLoader(dataset=Dataset('test'), batch_size=32, collate_fn=collate_fn,
                                              shuffle=True, drop_last=True)
    # 将下游任务模型切换到运行模式
    model.eval()
    correct = 0
    total = 0
    # 按批次遍历测试集中的数据
    for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(loader_test):
        # 计算5个批次，不全部遍历
        if i == 5:
            break
        print(i)
        # 不计算梯度
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # 统计正确率
        out = out.argmax(dim=1)  # 取最大值的位置
        correct += (out == labels).sum().item()  # 统计正确的个数
        total += len(labels)  # 统计总数
    print(correct / total)


if __name__ == '__main__':
    # 测试编码工具
    pretrained_model_name_or_path = snapshot_download('tiansz/bert-base-chinese')
    token = load_encode_tool(pretrained_model_name_or_path)
    # 加载训练数据集
    dataset = Dataset('train')
    # 定义计算设备
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    # 数据集加载器
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=16, collate_fn=collate_fn, shuffle=True,
                                         drop_last=True)
    # 加载预训练模型
    pretrained = BertModel.from_pretrained(Path(f'{pretrained_model_name_or_path}'))

    # 测试预训练模型
    pretrained.to(device)

    model = Model()
    model.to(device)

    train()
    test()
