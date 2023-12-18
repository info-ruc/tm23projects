import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 读取训练集和验证集数据
train_df = pd.read_csv('train.csv')
train_texts = train_df['text'].tolist()
train_labels = train_df['label'].tolist()

# 读取测试集数据
test_df = pd.read_csv('test.csv')
test_texts = test_df['text'].tolist()

# 加载BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 对训练集和测试集进行编码和划分
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

# 划分验证集
train_inputs, val_inputs, train_labels, val_labels = train_test_split(
    train_encodings['input_ids'],
    train_labels,
    test_size=0.2,
    random_state=42
)

train_masks, val_masks, _, _ = train_test_split(
    train_encodings['attention_mask'],
    train_labels,
    test_size=0.2,
    random_state=42
)

# 创建数据加载器
train_dataset = TensorDataset(torch.tensor(train_inputs),
                              torch.tensor(train_masks),
                              torch.tensor(train_labels))
val_dataset = TensorDataset(torch.tensor(val_inputs),
                            torch.tensor(val_masks),
                            torch.tensor(val_labels))
test_dataset = TensorDataset(torch.tensor(test_encodings['input_ids']),
                             torch.tensor(test_encodings['attention_mask']))

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 加载预训练的BERT模型
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model.to(device)

# 设置优化器和损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# 训练和验证
model.train()
for epoch in range(5):
    for batch in train_loader:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    # 验证
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for batch in val_loader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(outputs.logits, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print('Epoch:', epoch+1, 'Validation Accuracy:', accuracy)

# 测试
model.eval()
predictions = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        _, predicted = torch.max(outputs.logits, dim=1)
        predictions.extend(predicted.tolist())

# 将预测结果保存为CSV文件
test_df['label'] = predictions
test_df.to_csv('predictions.csv', index=False)