import random
import re
import string
import warnings

import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import BertForMaskedLM, BertModel, BertTokenizer

warnings.filterwarnings('ignore')

def filter_special_symbols(text, symbols=[]):
    try:
        if not isinstance(text, str):
            text = str(text)
        text = text.strip()
        for symbol in symbols:
            text = re.sub(symbol, "", text)
    except:
        pass
    return text


def RMSE(y, y_pred):
    with torch.no_grad():
        return torch.sqrt(((y - y_pred)**2).mean())


class PriceDataset(torch.utils.data.Dataset):
    def __init__(self, tokens, labels):
        self.labels = labels
        self.texts = tokens

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y


class BertPriceModel(nn.Module):
    def __init__(self, bert_model, dropout=0.):
        super(BertPriceModel, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(dropout)
        self.reg_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, input_id, mask):
        # with torch.no_grad():
        _, pooled_output = self.bert(
            input_ids=input_id, attention_mask=mask, return_dict=False)
        out = self.dropout(pooled_output)
        out = self.reg_head(out)
        # out = self.sigmoid(out)
        return out


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_one_epoch(epoch, model, dataloader, criterion, optimizer, device):
    model.train()
    rmse_meter = AverageMeter()
    loss_meter = AverageMeter()
    len_dataloader = len(dataloader)
    last_idx = len_dataloader - 1
    for i, (inputs, labels) in enumerate(dataloader):
        b = labels.size(0)

        mask = inputs['attention_mask']
        inputs_ids = inputs['input_ids'].squeeze(1)
        labels = labels.float()   # convert float64 to float32
        if device.type == 'cuda':
            inputs_ids = inputs_ids.cuda()
            mask = mask.cuda()
            labels = labels.cuda()
        outputs = model(inputs_ids, mask)
        batch_loss = criterion(outputs.squeeze(1), labels)
        # print(outputs.dtype, labels.dtype)
        # print(outputs[:2].cpu(), labels[:2].cpu(), batch_loss.item())

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        rmse = RMSE(labels, outputs)
        loss_meter.update(batch_loss.item(), b)
        rmse_meter.update(rmse.item(), b)
        if i % 100 == 0:
            msg = f"Train: {epoch} [{i:>4d}/{len_dataloader} ({100. * i / last_idx:>3.0f}%)] "\
                f"Loss: {loss_meter.val:.4g} ({loss_meter.avg:.3g}) "\
                f"RMSE: {rmse_meter.val:>7.4f} ({rmse_meter.avg:>7.4f})"
            print(msg)


def validate(epoch, model, dataloader, criterion, device):
    model.eval()
    rmse_meter = AverageMeter()
    loss_meter = AverageMeter()
    len_dataloader = len(dataloader)
    last_idx = len_dataloader - 1
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            b = labels.size(0)

            mask = inputs['attention_mask']
            inputs_ids = inputs['input_ids'].squeeze(1)
            labels = labels.float()   # convert float64 to float32
            if device.type == 'cuda':
                inputs_ids = inputs_ids.cuda()
                mask = mask.cuda()
                labels = labels.cuda()
            outputs = model(inputs_ids, mask)
            batch_loss = criterion(outputs.squeeze(1), labels)

            rmse = RMSE(labels, outputs)
            loss_meter.update(batch_loss.item(), b)
            rmse_meter.update(rmse.item(), b)
            if i % 100 == 0:
                msg = f"Valid: {epoch} [{i:>4d}/{len_dataloader} ({100. * i / last_idx:>3.0f}%)] "\
                    f"Loss: {loss_meter.val:.4g} ({loss_meter.avg:.3g}) "\
                    f"RMSE: {rmse_meter.val:>7.4f} ({rmse_meter.avg:>7.4f})"
                print(msg)


def process_data(path):
    html_symbol = r"<[^>]+>"
    chinese_symbol = r"[%s]+" % "，。？！：；【】『』「」“”‘’！·～…、丨█★—"
    english_symbol = r"[%s]+" % string.punctuation
    # number_symbol = r"\d+"
    space = r"\s+"
    symbols = [html_symbol, chinese_symbol, english_symbol]

    # beijing_path = 'data/beijing_listings-2020.10.24.csv'
    df = pd.read_csv(path, encoding='utf-8')

    # process price
    df['prices'] = df.price.apply(lambda x: float(
        x.strip().replace(',', '').split('$')[-1]))
    # df['prices_norm'] = np.log10(df.prices) # 梯度易爆图或者梯度消失问题
    print(
        f'Price: min: {df.prices.min()}, max: {df.prices.max()}, mean: {df.prices.mean()}, std: {df.prices.std()}')
    df['prices_norm'] = (df.prices - df.prices.mean()) / df.prices.std()
    # process description
    df['new_description'] = df.description.apply(
        lambda x: filter_special_symbols(x, symbols))
    df.drop_duplicates(subset=['new_description'], inplace=True)
    df_data = df.loc[:, ['new_description', 'prices', 'prices_norm']]

    np.random.seed(42)
    df_train, df_val, df_test = np.split(df_data.sample(frac=1, random_state=42),
                                         [int(.8*len(df_data)), int(.9*len(df_data))])

    return df_train, df_val, df_test


def tokenizer_text(tokenizer, df):
    tokens = [
        tokenizer(text,
                  padding='max_length',
                  max_length=512,
                  truncation=True,
                  return_tensors="pt") for text in df['new_description']
    ]
    labels = [label for label in df['prices_norm']]

    return tokens, labels


class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, lr_start, lr_base, warm_up=0, T_max=10, cur=0):
        """
        Description:
            - get warmup consine lr scheduler

        Arguments:
            - optimizer: (torch.optim.*), torch optimizer
            - lr_start: (float), minimum learning rate
            - lr_base: (float), maximum learning rate
            - warm_up: (int),  warm_up epoch or iteration
            - T_max: (int), maximum epoch or iteration

        Example:
            <<< epochs = 100
            <<< warm_up = 5
            <<< cosine_lr = WarmupCosineLR(optimizer, 1e-9, 1e-3, warm_up, epochs)
            <<< lrs = []
            <<< for epoch in range(epochs):
            <<<     optimizer.step()
            <<<     lrs.append(optimizer.state_dict()['param_groups'][0]['lr'])
            <<<     cosine_lr.step()
            <<< plt.plot(lrs, color='r')
            <<< plt.show()

        """
        self.lr_start = lr_start
        self.lr_base = lr_base
        self.warm_up = warm_up
        self.T_max = T_max
        self.cur = cur    # current epoch or iteration
        self.lrs = cosine_scheduler(
            lr_base, lr_start, T_max, 1, warm_up, lr_start)
        self.lrs = self.lrs.tolist()
        self.lrs.append(lr_start)

        super().__init__(optimizer, -1)

    def get_lr(self):
        lr = self.lrs[self.cur]
        self.cur += 1
        return [lr for base_lr in self.base_lrs]

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += f'optimizer={self.optimizer.__class__.__name__}, '
        format_string += f'lr_start={self.lr_start}, '
        format_string += f'lr_base={self.lr_base}, '
        format_string += f'warm_up={self.warm_up}, '
        format_string += f'T_max={self.T_max}, '
        format_string += f'cur={self.cur-1}'
        format_string += ')'
        return format_string


def cosine_scheduler(lr_base, lr_final, epochs, niter_per_ep, warmup_epochs=0, lr_start=1e-5):
    warmup_schedule = np.array([])

    # Fix bug, if warmup epochs greater than epochs
    # set warmup_epochs = 0, or could be set to epochs
    if warmup_epochs > epochs:
        warmup_epochs = 0
    warmup_iters = warmup_epochs * niter_per_ep

    if warmup_epochs > 0:
        warmup_schedule = np.linspace(lr_start, lr_base, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = lr_final + 0.5 * \
        (lr_base - lr_final) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def seed_everything(seed: int = 42) -> None:
    """set up all random seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def main():
    seed_everything(42)
    BERT_PATH = 'pretrained/'
    beijing_path = 'data/beijing_listings-2020.10.24.csv'
    dropout = 0.25
    start_epoch = 0
    epochs = 20
    batch_size = 32
    warmup_lr = 1e-7
    warmup_epochs = 0
    lr = 5e-6
    log_freq = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df_train, df_valid, df_test = process_data(beijing_path)
    print(
        f'train data: {len(df_train)}, val: {len(df_valid)}, test: {len(df_test)}')

    bert_tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
    train_tokens, train_labels = tokenizer_text(bert_tokenizer, df_train)
    val_tokens, val_labels = tokenizer_text(bert_tokenizer, df_valid)
    test_tokens, test_labels = tokenizer_text(bert_tokenizer, df_test)

    bert_model = BertModel.from_pretrained(BERT_PATH)

    model = BertPriceModel(bert_model, dropout)
    if device.type == 'cuda':
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    optimizer_scheduler = WarmupCosineLR(
        optimizer, warmup_lr, lr, warmup_epochs, epochs, start_epoch)
    print(f'Load BERT model and criterion ready.')

    train, val = PriceDataset(train_tokens, train_labels), PriceDataset(
        val_tokens, val_labels)

    # DataLoader根据batch_size获取数据，训练时选择打乱样本
    train_dataloader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=batch_size)

    for epoch in range(epochs):
        train_one_epoch(epoch, model, train_dataloader,
                        criterion, optimizer, device)
        validate(epoch, model, val_dataloader, criterion, device)
        optimizer_scheduler.step()


if __name__ == "__main__":
    main()
