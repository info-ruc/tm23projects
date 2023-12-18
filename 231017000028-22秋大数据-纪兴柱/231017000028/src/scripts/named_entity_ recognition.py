#!/usr/bin/env python
# coding: utf-8


import torch
from transformers import AutoTokenizer
from datasets import load_dataset,load_from_disk
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification, DistilBertModel, PreTrainedModel, PretrainedConfig
from transformers import AdamW
from transformers.optimization import get_scheduler


#全局变量
load_from_local = True


def get_dataset():

    #加载本地已处理好的数据集
    if load_from_local:
        dataset = load_from_disk('./src/dataset/')
        return dataset
    
    #远程加载需要预处理
    dataset = load_dataset(path='conll2003')

    print('查看数据样例')
    print(dataset, dataset['train'][0])

    #数据处理函数
    def tokenize_and_align_labels(data):
        #分词
        data_encode = tokenizer.batch_encode_plus(data['tokens'],
                                                  truncation=True,
                                                  is_split_into_words=True)

        data_encode['labels'] = []
        for i in range(len(data['tokens'])):
            label = []
            for word_id in data_encode.word_ids(batch_index=i):
                if word_id is None:
                    label.append(-100)
                else:
                    label.append(data['ner_tags'][i][word_id])

            data_encode['labels'].append(label)

        return data_encode

    dataset = dataset.map(
        tokenize_and_align_labels,
        batched=True,
        batch_size=1000,
        num_proc=1,
        remove_columns=['id', 'tokens', 'pos_tags', 'chunk_tags', 'ner_tags'])

    return dataset


#定义下游任务模型
class Model(PreTrainedModel):
    config_class = PretrainedConfig

    def __init__(self, config):
        super().__init__(config)

        self.pretrained = DistilBertModel.from_pretrained(
            'distilbert-base-uncased')

        #9 = len(dataset['train'].features['ner_tags'].feature.names)
        self.fc = torch.nn.Sequential(torch.nn.Dropout(0.1),
                                      torch.nn.Linear(768, 9))

        #加载预训练模型的参数
        parameters = AutoModelForTokenClassification.from_pretrained(
            'distilbert-base-uncased', num_labels=9)
        self.fc[1].load_state_dict(parameters.classifier.state_dict())

        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        logits = self.pretrained(input_ids=input_ids,
                                 attention_mask=attention_mask)
        logits = logits.last_hidden_state

        logits = self.fc(logits)

        loss = None
        if labels is not None:
            loss = self.criterion(logits.flatten(end_dim=1), labels.flatten())

        return {'loss': loss, 'logits': logits}


#测试
def test():
    model.eval()

    #数据加载器
    loader_test = torch.utils.data.DataLoader(
        dataset=dataset['test'],
        batch_size=16,
        collate_fn=DataCollatorForTokenClassification(tokenizer),
        shuffle=True,
        drop_last=True,
    )

    labels = []
    outs = []
    for i, data in enumerate(loader_test):
        #计算
        with torch.no_grad():
            out = model(**data)

        out = out['logits'].argmax(dim=2)

        for j in range(16):
            select = data['attention_mask'][j] == 1
            labels.append(data['labels'][j][select][1:-1])
            outs.append(out[j][select][1:-1])

        if i % 10 == 0:
            print(i)

        if i == 50:
            break

    #计算正确率
    labels = torch.cat(labels)
    outs = torch.cat(outs)

    print((labels == outs).sum().item() / len(labels))


#训练
def train():
    optimizer = AdamW(model.parameters(), lr=2e-5)
    scheduler = get_scheduler(name='linear',
                              num_warmup_steps=0,
                              num_training_steps=len(loader),
                              optimizer=optimizer)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.train()
    model.to(device)
    for i, data in enumerate(loader):
        for k in data.keys():
            data[k] = data[k].to(device)
            
        out = model(**data)
        loss = out['loss']

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        optimizer.zero_grad()
        model.zero_grad()
        if i % 50 == 0:
            labels = []
            outs = []
            out = out['logits'].argmax(dim=2)
            for j in range(8):
                select = data['attention_mask'][j] == 1
                labels.append(data['labels'][j][select][1:-1])
                outs.append(out[j][select][1:-1])

            #计算正确率
            labels = torch.cat(labels)
            outs = torch.cat(outs)
            accuracy = (labels == outs).sum().item() / len(labels)

            lr = optimizer.state_dict()['param_groups'][0]['lr']

            print(i, loss.item(), accuracy, lr)

    model.to('cpu')



#加载编码器
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased',
                                        use_fast=True)
print(tokenizer)

dataset = get_dataset()

print(dataset, dataset['train'][0])

#数据加载器
loader = torch.utils.data.DataLoader(
    dataset=dataset['train'],
    batch_size=8,
    collate_fn=DataCollatorForTokenClassification(tokenizer),
    shuffle=True,
    drop_last=True,
)

for i, data in enumerate(loader):
    break

for k, v in data.items():
    print(k, v.shape, v[:2])

len(loader)

model = Model(PretrainedConfig())

#统计参数量
print(sum(i.numel() for i in model.parameters()) / 10000)

out = model(**data)

out['loss'], out['logits'].shape

#训练前测试
test()

#训练
train()

#训练前测试
test()

