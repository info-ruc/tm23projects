import pandas as pd
import torch
from torch.utils.data import DataLoader
import datasets
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from torchkeras import KerasModel
import evaluate

df = pd.read_csv("../dataset/waimai_10k.csv")
ds = datasets.Dataset.from_pandas(df)
ds = ds.shuffle(42) #打乱顺序
ds = ds.rename_columns({"review":"text","label":"labels"})

tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese') #需要和模型一致
if __name__ == '__main__':
    ds_encoded = ds.map(lambda example:tokenizer(example["text"],
                      max_length=50,truncation=True,padding='max_length'),
                      batched=True,
                      batch_size=20,
                      num_proc=2) #支持批处理和多进程map
    # 转换成pytorch中的tensor
    ds_encoded.set_format(type="torch", columns=["input_ids", 'attention_mask', 'token_type_ids', 'labels'])
    # 分割成训练集和测试集
    ds_train_val, ds_test = ds_encoded.train_test_split(test_size=0.2).values()
    ds_train, ds_val = ds_train_val.train_test_split(test_size=0.2).values()
    # 在collate_fn中可以做动态批处理(dynamic batching)
    def collate_fn(examples):
        return tokenizer.pad(examples)
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=16, collate_fn=collate_fn)
    dl_val = torch.utils.data.DataLoader(ds_val, batch_size=16, collate_fn=collate_fn)
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=16, collate_fn=collate_fn)
    for batch in dl_train:
        break
        # 加载模型 (会添加针对特定任务类型的Head)
    model = AutoModelForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2)
    dict(model.named_children()).keys()
    output = model(**batch)
    class StepRunner:
        def __init__(self, net, loss_fn, accelerator, stage="train", metrics_dict=None,
                     optimizer=None, lr_scheduler=None
                     ):
            self.net, self.loss_fn, self.metrics_dict, self.stage = net, loss_fn, metrics_dict, stage
            self.optimizer, self.lr_scheduler = optimizer, lr_scheduler
            self.accelerator = accelerator
            if self.stage == 'train':
                self.net.train()
            else:
                self.net.eval()
        def __call__(self, batch):
            out = self.net(**batch)
            # loss
            loss = out.loss
            # preds
            preds = (out.logits).argmax(axis=1)
            # backward()
            if self.optimizer is not None and self.stage == "train":
                self.accelerator.backward(loss)
                self.optimizer.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                self.optimizer.zero_grad()
            all_loss = self.accelerator.gather(loss).sum()
            labels = batch['labels']
            acc = (preds == labels).sum() / ((labels > -1).sum())
            all_acc = self.accelerator.gather(acc).mean()
            # losses
            step_losses = {self.stage + "_loss": all_loss.item(), self.stage + '_acc': all_acc.item()}
            # metrics
            step_metrics = {}
            if self.stage == "train":
                if self.optimizer is not None:
                    step_metrics['lr'] = self.optimizer.state_dict()['param_groups'][0]['lr']
                else:
                    step_metrics['lr'] = 0.0
            return step_losses, step_metrics
    KerasModel.StepRunner = StepRunner
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    keras_model = KerasModel(model,
                             loss_fn=None,
                             optimizer=optimizer
                             )

    keras_model.fit(
        train_data=dl_train,
        val_data=dl_val,
        ckpt_path='bert_waimai.pt',
        epochs=100,
        patience=10,
        monitor="val_acc",
        mode="max",
        plot=True,
        wandb=False,
        quiet=True
    )
    model.eval()
    model.config.id2label = {0: "差评", 1: "好评"}
    model.save_pretrained("waimai_10k_bert")
    tokenizer.save_pretrained("waimai_10k_bert")
