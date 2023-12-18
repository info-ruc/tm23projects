import torch
import torch.nn as nn
import random
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, BertModel, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
SEED = 1234
batch_size = 16
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


class Model(nn.Module):
    def __init__(self, pretrained_weights='bert-base-chinese'):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_weights)
        self.cls = nn.Linear(768, 5)

    def forward(self, input_ids, token_type_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        output = self.cls(bert_output[1])
        return output


def text_encoder(tokenizer, dataset, max_len=150):
    # 将文本转换为编码
    text_list, labels = dataset['comment_processed'].values.tolist(), dataset['label'].values
    tokenizer = tokenizer(text_list, padding=True, truncation=True, max_length=max_len, return_tensors='pt')
    input_ids, token_type_ids, attention_mask = tokenizer['input_ids'], tokenizer['token_type_ids'], tokenizer[
        'attention_mask']
    labels = torch.from_numpy(labels)
    data = TensorDataset(input_ids, token_type_ids, attention_mask, labels)
    return data


def train(model, train_loader, eval_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-5, eps=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, min_lr=1e-7, patience=5, verbose=True,
                                  threshold=0.0001, eps=1e-08)
    t_total, total_epochs, bestAcc, correct, total = len(train_loader), 2, 0, 0, 0
    print('start training')
    for epoch in range(total_epochs):
        for step, (input_ids, token_type_ids, attention_mask, labels) in enumerate(train_loader):
            input_ids, token_type_ids, attention_mask, labels = (input_ids.cuda(), token_type_ids.cuda(),
                                                                 attention_mask.cuda(), labels.cuda())
            with torch.cuda.amp.autocast(enabled=True):
                out_put = model(input_ids.cuda(), token_type_ids.cuda(), attention_mask.cuda())
                loss = criterion(out_put, labels)

            total += labels.size(0)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if (step + 1) % 2 == 0:
                _, predict = torch.max(out_put.data, 1)
                correct += (predict == labels).sum().cpu().numpy()
                train_acc = correct / total
                print(f"Train epoch[{epoch + 1}/{total_epochs}],step[{step + 1}/{len(train_loader)}],"
                      f"tra_acc:{round(train_acc * 100, 2)} %,loss:{np.round(loss.detach().cpu().numpy(), 3)}")
            if (step + 1) % 50 == 0:
                acc = eval(model, eval_loader)
                model.train()
                if bestAcc < acc:
                    bestAcc = acc
                    torch.save(model.state_dict(), 'savedmodel/best.pkl')
                print(f"Eval epoch[{epoch + 1}/{total_epochs}],step[{step + 1}/{len(train_loader)}],"
                      f"tra_acc:{round(train_acc * 100, 3)} %,bestAcc:{round(bestAcc * 100, 3)}%,"
                      f"eval_acc:{round(acc * 100, 3)} %,loss:{np.round(loss.detach().cpu().numpy(), 3)}")
        scheduler.step(bestAcc)


def eval(model, eval_loader):
    model.eval()
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True):
            correct = 0
            total = 0
            for step, (input_ids, token_type_ids, attention_mask, labels) in tqdm(enumerate(eval_loader),
                                                                                  desc='Dev Itreation:'):
                input_ids, token_type_ids, attention_mask, labels = input_ids.cuda(), token_type_ids.cuda(), attention_mask.cuda(), labels.cuda()
                out_put = model(input_ids, token_type_ids, attention_mask)
                _, predict = torch.max(out_put.data, 1)
                correct += (predict == labels).sum().cpu().numpy()
                total += labels.size(0)
            res = correct / total
    return res


def main():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    data = pd.read_csv("data.csv")
    train_dataset, test_dataset = train_test_split(data, test_size=0.2, random_state=0)
    train_data, eval_data = text_encoder(tokenizer, train_dataset), text_encoder(tokenizer, test_dataset)

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(dataset=eval_data, batch_size=batch_size, shuffle=False)

    model = Model()
    model.cuda()
    train(model, train_loader, eval_loader)


if __name__ == '__main__':
    main()
