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

        for param in self.bert.parameters():
            param.requires_grad = True
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

def load_checkpoint(checkpoint, model=None, strict=False):
    if hasattr(model, 'module'):
        model = model.module
    device = torch.cuda.current_device()
    src_state_dict = torch.load(checkpoint, map_location=lambda storage, loc: storage.cuda(device))
    target_state_dict = model.state_dict()
    skip_keys = []
    # skip mismatch size tensors in case of pretraining
    for k in src_state_dict.keys():
        if k not in target_state_dict:
            continue
        if src_state_dict[k].size() != target_state_dict[k].size():
            skip_keys.append(k)
    for k in skip_keys:
        del src_state_dict[k]
    missing_keys, unexpected_keys = model.load_state_dict(src_state_dict, strict=strict)
    if skip_keys:
        print(
            f'removed keys in source state_dict due to size mismatch: {", ".join(skip_keys)}')
    if missing_keys:
        print(f'missing keys in source state_dict: {", ".join(missing_keys)}')
    if unexpected_keys:
        print(f'unexpected key in source state_dict: {", ".join(unexpected_keys)}')

def main():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    data = pd.read_csv("data.csv")
    train_dataset, test_dataset = train_test_split(data, test_size=0.2, random_state=0)
    train_data, eval_data = text_encoder(tokenizer, train_dataset), text_encoder(tokenizer, test_dataset)
    eval_loader = DataLoader(dataset=eval_data, batch_size=batch_size, shuffle=False)

    model = Model()
    load_checkpoint('savedmodel/best.pkl', model)
    model.cuda()
    acc = eval(model, eval_loader)
    print(f'eval_acc:{round(acc * 100, 3)} %')

if __name__ == '__main__':
    main()
