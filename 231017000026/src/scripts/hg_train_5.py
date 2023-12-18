# -*- coding:GB2312 -*-
# %%
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('hfl/rbt3')

tokenizer.batch_encode_plus(
['一曲青溪一曲山', '鸟飞鱼跃白云间'],
truncation=True,
)

# %%
#第6章/从磁盘加载数据集
from datasets import load_from_disk, Dataset
# dataset = load_from_disk('./data/ChnSentiCorp/')
# print(type(dataset), dataset[0:50])

dataset_train = Dataset.from_file('./data/chn_senti_corp/chn_senti_corp-train.arrow')
dataset_test = Dataset.from_file('./data/chn_senti_corp/chn_senti_corp-test.arrow')
dataset_valid = Dataset.from_file('./data/chn_senti_corp/chn_senti_corp-validation.arrow')
print(type(dataset_train), dataset_train[0:50])

# %%
# 缩小数据规模，便于测试
dataset_train= dataset_train.shuffle().select(range(3000))
dataset_test= dataset_test.shuffle().select(range(200))
print(type(dataset_train))

# %%
#第6章/编码
def f(data):
    return tokenizer.batch_encode_plus(data['text'],truncation=True)

dataset_train=dataset_train.map(f,
batched=True,
batch_size=100,
# num_proc=4,
remove_columns=['text'])

print(dataset_train)
# %%
print(dataset_test)
dataset_test=dataset_test.map(f,
batched=True,
batch_size=100,
# num_proc=4,
remove_columns=['text'])

# %%
print(type(dataset_test), len(dataset_test), dataset_test)
# %%
def filter_func(data):
    return [len(i)<=512 for i in data['input_ids']]

dataset_train=dataset_train.filter(filter_func, batched=True, batch_size=100)

dataset_test=dataset_test.filter(filter_func, batched=True, batch_size=100)

print(type(dataset_train), len(dataset_train), dataset_train)
print(type(dataset_test), len(dataset_test), dataset_test)

# %%
from transformers import AutoModelForSequenceClassification
import torch
model=AutoModelForSequenceClassification.from_pretrained('hfl/rbt3',num_labels=2)
# #统计模型参数量
# sum([i.nelement() for i in model.parameters()]) / 10000
# # %%
# #第6章/模型试算
# #模拟一批数据
# data = {
# 'input_ids': torch.ones(40, 100, dtype=torch.long),
# 'token_type_ids': torch.ones(40, 100, dtype=torch.long),
# 'attention_mask': torch.ones(40, 100, dtype=torch.long),
# 'labels': torch.ones(40, dtype=torch.long)
# }#模型试算

# out = model(**data)
# out['loss'], out['logits'].shape
# %%
#第6章/加载评价指标
from datasets import load_metric
metric = load_metric('accuracy')

#第6章/定义评价函数
import numpy as np
from transformers.trainer_utils import EvalPrediction
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits = logits.argmax(axis=1)
    return metric.compute(predictions=logits, references=labels)

#模拟输出
eval_pred = EvalPrediction(
predictions=np.array([[0, 1], [2, 3], [4, 5], [6, 7]]),
label_ids=np.array([1, 1, 0, 1]),
)
# %%
print("1 >> ", compute_metrics(eval_pred))
print(eval_pred.predictions.shape, eval_pred.label_ids.shape)
# %%
#第6章/定义训练参数
from transformers import TrainingArguments
#定义训练参数
args = TrainingArguments(
#定义临时数据保存路径
output_dir='./output_dir/third/',
#定义测试执行的策略，可取值为no、epoch、steps
evaluation_strategy='steps',
#定义每隔多少个step执行一次测试
eval_steps=30,
#定义模型保存策略，可取值为no、epoch、steps
save_strategy='steps',
#定义每隔多少个step保存一次
save_steps=30,
#定义共训练几个轮次
num_train_epochs=2,
learning_rate=1e-4,#定义学习率
#加入参数权重衰减，防止过拟合
weight_decay=1e-2,
#定义测试和训练时的批次大小
per_device_eval_batch_size=16,
per_device_train_batch_size=16,
#定义是否要使用GPU训练
no_cuda=False,
)
# %%
#第6章/定义训练器
from transformers import Trainer
from transformers.data.data_collator import DataCollatorWithPadding
#定义训练器
trainer = Trainer(
model=model,
args=args,
train_dataset=dataset_train,
eval_dataset=dataset_test,
compute_metrics=compute_metrics,
data_collator=DataCollatorWithPadding(tokenizer),
)

# %%
#第6章/测试数据整理函数
data_collator = DataCollatorWithPadding(tokenizer)
#获取一批数据
data = dataset_train[:5]
#输出这些句子的长度
for i in data['input_ids']:
    print(len(i))
#调用数据整理函数
data = data_collator(data)
#查看整理后的数据
for k, v in data.items():
    print(k, v.shape)
# %%
tokenizer.decode(data['input_ids'][0])
# %%
#评价模型
trainer.evaluate()

# %%
#第6章/训练
trainer.train()
# %%
trainer.evaluate()


# %%
trainer.save_model(output_dir='./output_dir/save_model_third')
# %%
# import torch
# print(torch.cuda.is_available())
# device = torch.device("cuda:0")
# print(device)
# %%
# trainer.train(resume_from_checkpoint='./output_dir/checkpoint-90')

# %%
