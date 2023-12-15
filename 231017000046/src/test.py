#
# @Author: wangdong
# @Email:  whitewdxx@gmail.com
#
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.metrics import accuracy_score, classification_report
import os

def read_imdb(data_dir, is_train):
    data, labels = [], []
    for label in ('pos', 'neg'):
        folder_name = os.path.join(data_dir, 'train' if is_train else 'test',
                                   label)
        for file in os.listdir(folder_name):
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '')
                data.append(review)
                labels.append(1 if label == 'pos' else 0)
    return data, labels

# 加载模型和分词器
model = BertForSequenceClassification.from_pretrained('./sentiment_model_final')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 你的测试文本
test_text = "This is a positive movie review."

# 对测试文本进行分词和编码
inputs = tokenizer(test_text, return_tensors="pt")

# 在模型上进行推理
with torch.no_grad():
    outputs = model(**inputs)

# 获取预测的标签
predicted_label = torch.argmax(outputs.logits).item()

# 打印预测结果
print(f"Predicted Label: {predicted_label}")

test_texts, test_labels = read_imdb('../dataset/aclImdb_v1/aclImdb',False)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128, return_tensors='pt')
test_dataset = torch.utils.data.TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], torch.tensor(test_labels))
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

predictions = []
with torch.no_grad():
    for batch in test_loader:
        inputs, attention_mask, labels = batch
        outputs = model(inputs, attention_mask=attention_mask)
        logits = outputs.logits
        predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
# 计算准确率和其他指标
accuracy = accuracy_score(test_labels, predictions)
report = classification_report(test_labels, predictions)
print(f'Accuracy: {accuracy}')
print('Classification Report:\n', report)
