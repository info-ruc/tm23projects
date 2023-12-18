#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# https://zhuanlan.zhihu.com/p/535133036 参考资料 Hugging Face-Transformers学习笔记（二）——情感分类
# 因为无python基础，基本上是面向ai编程，大部分工作是把代码copy下来进行实验
# 预计在作业提交之后，还会花多一些时间进行代码上的改动，或者是项目上的调整


# In[3]:


# 列出hugging face的数据集，并查看前十条的名称
from datasets import list_datasets
all_datasets = list_datasets()
print(f"There are {len(all_datasets)} datasets currently available on the Hub")
print(f"The first 10 are: {all_datasets[:10]}")


# In[4]:


# 加载emotions数据集
from datasets import load_dataset
emotions = load_dataset("emotion")
emotions


# In[5]:


print(train_ds["text"][:5])


# In[11]:


from datasets import Dataset, load_csv, load_image_files
from torch.utils.data import DataLoader

# Load the dataset using the load_csv function
train_ds = load_csv('C:\Users\zhouy134\Documents\应用\homework\tm23projects\231017000188\to\train.csv')

# Load the images using the load_image_files function
train_images = load_image_files('C:\Users\zhouy134\Documents\应用\homework\tm23projects\231017000188/to/train_images', transform=transforms.Resize((224, 224)))

# Create a DataLoader using the dataset
train_dataloader = DataLoader(train_ds, batch_size=32, shuffle=True)

# Access the data using the dataset
print(train_ds["text"][:5])


# In[6]:


# From Text to Tokens
# 将文本转化为标记是非常重要的一个过程，最基础的是字符标记化，将每一个字母映射到一个数字
text = "Tokenizing text is a core task of NLP."
tokenized_text = list(text)
print(tokenized_text)

token2idx = {ch: idx for idx, ch in enumerate(sorted(set(tokenized_text)))}
print(token2idx)


import torch
import torch.nn.functional as F
input_ids = torch.tensor(input_ids)
one_hot_encodings = F.one_hot(input_ids, num_classes=len(token2idx))
one_hot_encodings.shape


# In[12]:


tokenized_text = text.split()
print(tokenized_text)


# In[13]:


from transformers import AutoTokenizer
model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)


# In[14]:


def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)


# In[15]:


from transformers import AutoModel
model_ckpt = "distilbert-base-uncased"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained(model_ckpt).to(device)


# In[16]:


# Training a Text Classifier
# Feature Extractors方法
# 简单复习一下特征提取的方法，pretrained（）方法加载预训练模型的权重，加载需要用到的模型。

from transformers import AutoModel
model_ckpt = "distilbert-base-uncased"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained(model_ckpt).to(device)


# In[17]:


#接下来是提取隐藏状态：模型只返回一个属性，即最后一个隐藏状态，对于分类任务，通常只使用与[CLS]标记关联的隐藏状态作为输入特性。由于该标记出现在每个序列的开头，我们可以通过简单地将其索引到输出中来提取它。具体步骤为，先定义一个提取隐藏状态的函数：

def extract_hidden_states(batch):
    # Place model inputs on the GPU
    inputs = {k:v.to(device) for k,v in batch.items()if k in tokenizer.model_input_names}
    # 提取隐藏状态
    with torch.no_grad():
        last_hidden_state = model(**inputs).last_hidden_state
    # Return vector for [CLS] token
    return {"hidden_state": last_hidden_state[:,0].cpu().numpy()}


# In[ ]:


#由于模型期望张量作为输入，接下来要做的是将input_ ID和attention_mask列转换为“torch”格式

emotions_encoded.set_format("torch",
                            columns=["input_ids", "attention_mask", "label"])


# In[ ]:


# 接下来提取隐藏状态：

emotions_hidden = emotions_encoded.map(extract_hidden_states, batched=True)


# In[ ]:


# 此时为数据集 添加了hidden_state 这一列

emotions_hidden["train"].column_names
['attention_mask', 'hidden_state', 'input_ids', 'label', 'text']


# In[ ]:


#现在有了与每句话相关的隐藏状态，下一步就是在它们上训练分类器。这里使用了DummyClassifier，可以使用简单的启发式方法构建分类器，例如总是选择多数类或总是绘制随机类。在这种情况下，表现最好的启发式方法是总是选择最频繁的类，这会产生大约35%的准确率：

from sklearn.dummy import DummyClassifier
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, y_train)
dummy_clf.score(X_valid, y_valid)
0.352


# In[18]:


# Fine-Tuning方法
#Fine-Tuning方法使得整个DistilBERT模型将与分类头一起训练，下面整个回顾一下使用Fine-Tuning方法的具体步骤：
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[ ]:


# 在整个数据集上进行标记化：

from transformers import AutoTokenizer
model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)


# In[ ]:


# 使用端到端的训练方法，加载数据：

from transformers import AutoModelForSequenceClassification
num_labels = 6
model = (AutoModelForSequenceClassification
         .from_pretrained(model_ckpt, num_labels=num_labels)
         .to(device))


# In[ ]:


# 定义性能指标进行误差分析：

from sklearn.metrics import accuracy_score, f1_score
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}


# In[ ]:


# 将模型保存到hugging face平台，方便以后直接调用：

from huggingface_hub import notebook_login
notebook_login()


# In[ ]:


# 训练模型：

from transformers import Trainer, TrainingArguments
batch_size = 64
logging_steps = len(emotions_encoded["train"]) // batch_size
model_name = f"{model_ckpt}-finetuned-emotion"
training_args = TrainingArguments(output_dir=model_name,
                  num_train_epochs=2,
                  learning_rate=2e-5,
                  per_device_train_batch_size=batch_size,
                  per_device_eval_batch_size=batch_size,
                  weight_decay=0.01,
                  evaluation_strategy="epoch",
                  disable_tqdm=False,
                  logging_steps=logging_steps,
                  push_to_hub=True,
                  log_level="error")
from transformers import Trainer
trainer = Trainer(model=model, args=training_args,
                  compute_metrics=compute_metrics,
                  train_dataset=emotions_encoded["train"],
                  eval_dataset=emotions_encoded["validation"],
                  tokenizer=tokenizer)
trainer.train();


# In[ ]:


# 定义误差分析需要的函数：

def label_int2str(row):
    return emotions["train"].features["label"].int2str(row)

from torch.nn.functional import cross_entropy
def forward_pass_with_label(batch):
    # Place all input tensors on the same device as the model
    inputs = {k:v.to(device) for k,v in batch.items()if k in tokenizer.model_input_names}
    with torch.no_grad():
        output = model(**inputs)
        pred_label = torch.argmax(output.logits, axis=-1)
        loss = cross_entropy(output.logits, batch["label"].to(device),
                              reduction="none")
        # Place outputs on CPU for compatibility with other dataset columns
    return{"loss": loss.cpu().numpy(),
                "predicted_label": pred_label.cpu().numpy()}
# Convert our dataset back to PyTorch tensors
emotions_encoded.set_format("torch",
                            columns=["input_ids", "attention_mask", "label"])
# Compute loss values
emotions_encoded["validation"] = emotions_encoded["validation"].map(
    forward_pass_with_label, batched=True, batch_size=16)
emotions_encoded.set_format("pandas")
cols = ["text", "label", "predicted_label", "loss"]
df_test = emotions_encoded["validation"][:][cols]
df_test["label"] = df_test["label"].apply(label_int2str)
df_test["predicted_label"] = (df_test["predicted_label"].apply(label_int2str))


# In[ ]:


# 查看误差最高的10句和误差最低的10句：

df_test.sort_values("loss", ascending=False).head(10)

df_test.sort_values("loss", ascending=True).head(10)


# In[ ]:


# 保存模型到hugging face平台：

trainer.push_to_hub(commit_message="Training completed!")

from transformers import pipeline

#Eleven是我的hub useename
model_id = "Eleven/distilbert-base-uncased-finetuned-emotion"
classifier = pipeline("text-classification", model=model_id)

