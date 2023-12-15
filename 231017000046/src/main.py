#
# @Author: wangdong
# @Email:  whitewdxx@gmail.com
#
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os
from transformers import default_data_collator

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

def custom_data_collator(features):
    if isinstance(features[0], dict):
        # When features is a list of dictionaries
        return default_data_collator(features)
    elif isinstance(features[0], tuple):
        # When features is a list of tuples
        input_ids = torch.stack([torch.tensor(f[0]) for f in features])
        attention_mask = torch.stack([torch.tensor(f[1]) for f in features])
        labels = torch.tensor([f[2] for f in features])
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    else:
        raise ValueError("Unsupported features format")

# 加载IMDb数据集
# dataset = load_dataset("imdb")

# 划分训练集和测试集
# train_texts, test_texts, train_labels, test_labels = train_test_split(dataset["train"]["text"], dataset["train"]["label"], test_size=0.2, random_state=42)

# 读取训练集和测试集 从官网下载后本地读取     https://ai.stanford.edu/~amaas/data/sentiment/
train_texts, train_labels = read_imdb('../dataset/aclImdb_v1/aclImdb',True)
test_texts, test_labels = read_imdb('../dataset/aclImdb_v1/aclImdb',False)

# 加载BERT的分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 将文本转换为BERT的输入格式
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128, return_tensors='pt')
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128, return_tensors='pt')

# 创建PyTorch数据集
train_dataset = torch.utils.data.TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], torch.tensor(train_labels))
test_dataset = torch.utils.data.TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], torch.tensor(test_labels))

# 加载BERT模型
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# 移动模型到GPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# 定义训练参数
training_args = TrainingArguments(
    output_dir="./sentiment_model",  # 保存模型的文件夹
    num_train_epochs=3,               # 训练轮数
    per_device_train_batch_size=8,    # 每个设备上的训练批次大小
    save_steps=1000,                  # 每1000步保存一次模型
    save_total_limit=2,               # 最多保存2个模型
    logging_dir="./logs",             # 日志输出文件夹
    logging_steps=500,                # 每500步输出一次日志
    evaluation_strategy="steps",      # 在每500步后进行评估
    eval_steps=500,                   # 每500步进行一次评估
    logging_first_step=True,
    load_best_model_at_end=True,
)

# # 定义Trainer对象
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=test_dataset,
#     compute_metrics=lambda p: {"accuracy": accuracy_score(p.predictions.argmax(axis=1), p.label_ids)}
# )

# 在Trainer中使用新的数据collator
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=custom_data_collator,  # 使用自定义的数据collator
    compute_metrics=lambda p: {"accuracy": accuracy_score(p.predictions.argmax(axis=1), p.label_ids)}
)

# 开始训练
trainer.train()

# 保存最终模型
trainer.save_model("./sentiment_model_final")

# 在测试集上评估最终模型
results = trainer.evaluate()

# 打印结果
print(results)
