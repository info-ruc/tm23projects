from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    TextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    pipeline,
)
from datasets import load_dataset
import math

model_checkpoint = "bert-base-chinese"

# 加载预训练的分词器
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# 加载预训练的模型
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)

# 加载数据集
dataset = load_dataset("json", data_files="./train.json")
print("加载数据集", dataset)


# 处理数据集
def tokenize_function(examples):
    result = tokenizer(examples["text"])
    result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result


# 处理数据为 包含 input_ids、token_type_ids、attention_mask、 word_ids指定格式
lm_datasets = dataset.map(
    tokenize_function, batched=True, remove_columns=["text", "label"]
)
print("处理后数据集", lm_datasets)
print("随便取一条数据", tokenizer.decode(lm_datasets["train"][1]["input_ids"]))


# 定义训练集和测试集

# 60%用于训练
train_size = int(min(500, len(lm_datasets["train"]) * 0.6))

# 40%用于测试
test_size = int(0.4 * train_size)

# 构造样本集
sampled_dataset = lm_datasets["train"].train_test_split(
    train_size=train_size, test_size=test_size, seed=42
)

print("sampled_dataset", sampled_dataset["train"])

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm_probability=0.4
)

# 查看2条数据处理格式，可以看到部分字词被[MASK]所掩盖占位
samples = [sampled_dataset["train"][i] for i in range(2)]
print("samples", samples)
for sample in samples:
    _ = sample.pop("word_ids")

for chunk in data_collator(samples)["input_ids"]:
    print(f"\n'>>> {tokenizer.decode(chunk)}'")

# 定义训练器

# 样本很小，我们这边设置 每次投喂的数据为2条
batch_size = 2
logging_steps = int(len(sampled_dataset["train"]) / batch_size)
print("logging_steps", logging_steps, len(sampled_dataset["train"]))

training_args = TrainingArguments(
    output_dir="./output",
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    num_train_epochs=10,
    learning_rate=5e-6,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    push_to_hub=False,
    fp16=False,
    logging_steps=logging_steps,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=sampled_dataset["train"],
    eval_dataset=sampled_dataset["test"],
    data_collator=data_collator,
)

# 训练前
eval_results = trainer.evaluate()
print(f">>> 训练前 loss: {math.exp(eval_results['eval_loss']):.2f}")
# 训练前预览
mask_filler = pipeline("fill-mask", model=model_checkpoint, top_k=20)
# step1、定义预测文本
text = "小孩的名字叫张[MASK]"
preds = mask_filler(text)
# step2、输出预测结果
for pred in preds:
    print(f">>> {pred['sequence']}")

# 触发训练
trainer.train()

# 训练后
eval_results = trainer.evaluate()
print("eval_results", eval_results)
print(f">>> 训练后 loss: {math.exp(eval_results['eval_loss']):.2f}")

# 保存模型
import os

os.chdir("./")
tokenizer.save_pretrained("./new-model")
model.save_pretrained("./new-model")
print("模型保存成功")


# 训练后预览
mask_filler2 = pipeline(
    "fill-mask",
    model="./new-model",
    top_k=20,
)
preds2 = mask_filler2("小孩的名字叫张[MASK]")
for pred in preds2:
    print(f">>> {pred['sequence']}")
