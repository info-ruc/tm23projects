from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
import tempfile
import json
from modelscope.msdatasets import MsDataset
from datasets import Dataset
from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer

#当前模型为PALM 2.0摘要生成模型-中文-large
model_name = 'damo/nlp_palm2.0_text-generation_chinese-large'


# step1: 首次执行语句
print('step1: 首次执行语句')
with open("test.json", "r") as f:
    input_data = json.load(f)
input = input_data[0]['source']

text_summary = pipeline(Tasks.text_generation, model=model_name)
result = text_summary(input)

# print('输入文本:\n' + input + '\n')
print('文本摘要结果:\n' + result[OutputKeys.TEXT])



# step2: 训练数据集
with open("data.json", "r") as f:
    train_data = json.load(f)

# 用自己数据集构造
train_dataset = MsDataset(Dataset.from_dict(train_data))
eval_dataset = MsDataset(Dataset.from_dict(train_data))

max_epochs = 10

num_warmup_steps = 100   #原始值500
def noam_lambda(current_step: int):
    current_step += 1
    return min(current_step**(-0.5),
               current_step * num_warmup_steps**(-1.5))

# 可以在代码修改 configuration 的配置
def cfg_modify_fn(cfg):
    cfg.preprocessor.sequence_length = 128
    cfg.train.lr_scheduler = {
        'type': 'LambdaLR',
        'lr_lambda': noam_lambda,
        'options': {
            'by_epoch': False
        }
    }
    cfg.train.optimizer = {
        "type": "AdamW",
        "lr": 1e-3,
        "options": {}
    }
    cfg.train.max_epochs = max_epochs   #原值15
    cfg.train.dataloader = {
        "batch_size_per_gpu": 8,
        "workers_per_gpu": 1
    }
    return cfg

work_dir = tempfile.TemporaryDirectory().name

kwargs = dict(
    model=model_name,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    work_dir=work_dir, #tempfile.TemporaryDirectory().name,
    cfg_modify_fn=cfg_modify_fn)
trainer = build_trainer(
    name=Trainers.text_generation_trainer, default_args=kwargs)
trainer.train()

print("####训练结束!")
print("训练路径： " + tmp_dir)


# step3: 再次执行测试语句
print('step3: 再次执行语句')

text_summary = pipeline(Tasks.text_generation, model=work_dir + "/output")
result = text_summary(input)

print('二次文本摘要结果:\n' + result[OutputKeys.TEXT])