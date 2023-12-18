import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import XLNetTokenizer, XLNetModel
from transformers import AutoTokenizer

# self.tokenizer = AutoTokenizer.from_pretrained('xlnet-base-cased')

# 定义自定义数据集类
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained('xlnet-base-cased')

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]

        encoded_text = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,
            pad_to_max_length=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'ids': encoded_text['input_ids'].squeeze(),
            'mask': encoded_text['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 定义XLNet和CNN模型
class XLNetCNN(nn.Module):
    def __init__(self, n_filters, filter_sizes, output_dim, dropout):
        super(XLNetCNN, self).__init__()
        self.xlnet = XLNetModel.from_pretrained('xlnet-base-cased')
        self.convs = nn.ModuleList([
            nn.Conv2d(1, n_filters, (fs, self.xlnet.config.hidden_size)) for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, ids, mask):
        x = self.xlnet(ids, attention_mask=mask)[0]
        x = x.unsqueeze(1)
        x = [nn.functional.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [nn.functional.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in x]
        x = torch.cat(x, dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# 读取训练集和测试集数据
train_data = pd.read_csv('chnsenticorp/train.tsv', delimiter='\t')
test_data = pd.read_csv('chnsenticorp/test.tsv', delimiter='\t')

# 分离标签和内容
train_texts = train_data['text_a'].tolist()
train_labels = train_data['label'].tolist()

test_texts = test_data['text_a'].tolist()
test_labels = test_data['label'].tolist()

# 划分训练集和验证集
train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=0.2, random_state=42)

# 创建TextDataset和DataLoader
train_dataset = TextDataset(train_texts, train_labels)
val_dataset = TextDataset(val_texts, val_labels)
test_dataset = TextDataset(test_texts, test_labels)

batch_size = 16

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

# 定义模型参数
output_dim = 2  # 分类数目
n_filters = 100  # 滤波器的数量
filter_sizes = [2, 3, 4]  # 卷积核的大小
dropout = 0.5

# 实例化模型
model = XLNetCNN(n_filters, filter_sizes, output_dim, dropout)

# 定义设备
# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
device = 'cuda:1'
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# 训练模型
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0

    for batch in train_dataloader:
        ids = batch['ids'].to(device)
        mask = batch['mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        predictions = model(ids, mask)
        loss = criterion(predictions, labels)
        train_loss += loss.item()

        _, predicted_labels = torch.max(predictions, 1)
        train_correct += (predicted_labels == labels).sum().item()

        loss.backward()
        optimizer.step()

    train_loss /= len(train_dataloader)
    train_accuracy = train_correct / len(train_dataset)

    # 在验证集上评估模型
    model.eval()
    val_loss = 0.0
    val_correct = 0

    with torch.no_grad():
        for batch in val_dataloader:
            ids = batch['ids'].to(device)
            mask = batch['mask'].to(device)
            labels = batch['labels'].to(device)

            predictions = model(ids, mask)
            loss = criterion(predictions, labels)
            val_loss += loss.item()

            _, predicted_labels = torch.max(predictions, 1)
            val_correct += (predicted_labels == labels).sum().item()

    val_loss /= len(val_dataloader)
    val_accuracy = val_correct / len(val_dataset)

    print(f'Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

    # 保存模型权重
    torch.save(model.state_dict(), f'model_weights_epoch{epoch+1}.pt')

# 在测试集上评估模型
model.eval()
test_correct = 0

with torch.no_grad():
    for batch in test_dataloader:
        ids = batch['ids'].to(device)
        mask = batch['mask'].to(device)
        labels = batch['labels'].to(device)

        predictions = model(ids, mask)

        _, predicted_labels = torch.max(predictions, 1)
        test_correct += (predicted_labels == labels).sum().item()

test_accuracy = test_correct / len(test_dataset)
print(f'Test Accuracy: {test_accuracy:.4f}')