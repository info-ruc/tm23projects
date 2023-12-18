# 本项目中由于引用了Hugging Face，运行时需要梯子，否则可能会因无法下载初始模型，而无法运行

# 依赖项命令：
# pip install -qqq bitsandbytes
# pip install -qqq datasets
# !pip install-qqq git+https://github.com/huggingface/transformers@de9255de27abfcae4a 1f816b904915f0b1e2
# !pip install-qqq git+https://github.com/huggingface/peft.git
# !pip install -qqq git+https://github.com/huggingface/accelerate.git
# !pip install -qqq einops
# !pip install -qqq scipy
# # 我们需要这个特定版本的transformers，因为当前的主分支存在一个bug，导致无法成功训练:
# !pip install git+https://github.com/huggingface/transformers@de9255de27abfcae4a 1f816b904915f0b1e2

# Hugging Face如果登不了，也可以试试它的镜像网站（https://hf-mirror.com/），通过它来下载初始模型。
# 初始模型也可以去ModelScope社区找一个(不过ModelScope的依赖环境，想在自己电脑上搭建起来，也挺花时间的)
# 或执行这行代码，直接安装 git clone https://hf-mirror.com/tiiuae/falcon-7b
import bitsandbytes as bab
import torch
import torch.nn as nn
# print("是否可用：", torch.cuda.is_available())        # 查看GPU是否可用
# print("GPU数量：", torch.cuda.device_count())        # 查看GPU数量
# print("torch方法查看CUDA版本：", torch.version.cuda)  # torch方法查看CUDA版本
# print("GPU索引号：", torch.cuda.current_device())    # 查看GPU索引号
# print("GPU名称：", torch.cuda.get_device_name(0))    # 根据索引号得到GPU名称

import transformers
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM


# 1、初始化模型：
model_name = "tiiuae/falcon-7b"
# 若已下载falcon-7b，则使用下面这行代码
model_name = "./falcon-7b"  #这里变量改成“model_path”更好
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map="auto"
)

# model_name：要使用的模型名称。请参考 Hugging Face 获取确切的模型名称。
# load_in_4bit：以 4 位量化模式加载模型，以便在有限的 GPU 内存下进行训练。使用 QLoRA 技术，我们不会损失模型性能。
# bnb_4bit_quant_type：fp4 或 nf4 之一。这设置了量化数据类型。nf4 是 QLoRA 特定的浮点类型，称为 NormalFloat。
# bnb_4bit_compute_dtype：这设置了计算类型，它可能与输入类型不同。例如，输入可能是 fp32，但计算可以设置为 bf16 来加速。对于 QLoRA 调优，请使用 bfloat16。
# trust_remote_code：为了加载 falcon 模型，需要执行一些 falcon 模型特定的代码（使其适合 transformer 接口）。涉及到的代码是configuration_RW.py和modelling_RW.py。
# device_map：定义将工作负载映射到哪个 GPU 设备上。设置为 auto 以最佳方式分配资源。


# 2、初始化分词器（负责从提示和响应中创建令牌的对象）：
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# AutoTokenizer：是一个Hugging Face Transformer Auto Class，它会根据模型架构自动创建正确的分词器实例
# pad_token：一个特殊令牌，用于将令牌数组调整为相同大小以便进行批处理。设置为句子结束（eos）令牌。

# 3、启用梯度检查点并调用准备方法：
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
# Gradient Checkpointing ：是一种用于在训练深度神经网络时减少内存占用的方法，代价是计算时间略有增加。
# 更多细节可以在这里找到：https://github.com/cybertronai/gradient-checkpointing
# prepare_model_for_kbit_training：这个方法包装了在运行参数高效微调训练（QLoRA是这种训练范式的一种方法）之前准备模型的整个协议。

# 4、初始化我们的LoRA训练配置：
config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["query_key_value"], lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, config)
# r, lora_alpha和lora_dropout是LoRA超参数。可以根据需要进行更改。对于 falcon 家族来说，上述参数通常是一个很好的起点。
# task_type：描述语言建模类型，是因果的还是掩码的。对于像 GPT、LLama 或 Falcon 这样的 Transformer，它是因果的。
# target_modules：变压器模型中应该用QLoRA进行训练的目标模块。实际上总是查询/键值模块。

# 5、加载数据集
# 加载JSON文件并将其转换为训练特征
data = load_dataset("json", data_files="dataset.json")


# 6、用 "<human>" 前缀标记我们的问题，用 "<assistant>" 前缀标记我们的答案。
def generate_prompt(question_answer):
    return """
<human>: {question_answer["question"]}
<assistant>: {question_answer["answer"]}
    """.strip()

def tokenize_prompt(question_answer):
    prompt = generate_prompt(question_answer)
    tokenized_prompt = tokenizer(prompt, padding=True, truncation=True)
    return tokenized_prompt

# 使用 shuffle 来重新排序列表，以消除潜在的排序偏差
data_prompt = data["train"].shuffle().map(tokenize_prompt)


# 配置训练器并运行模型训练。
# 下面的训练器对象参数是超参数，可以（也应该）根据需要进行更改以获得更好的结果。
# 对于 falcon-7b 模型训练，下面的参数非常有用
trainer = transformers.Trainer(
    model=model,
    train_dataset=data_prompt,
    args=transformers.TrainingArguments( per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    warmup_ratio=0.05,
    max_steps=80,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=1,
    output_dir="outputs",
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine"
), data_collator=transformers.DataCollatorForLanguageModeling(tokenizer,
mlm=False),)

model.config.use_cache = False
trainer.train()

# 在调整模型参数的时候，注意不要修改这个参数 optim="paged_adamw_8bit"，这是一种优化技术，可以避免内存峰值，从而防止 GPU 内存超载。
# max_steps 参数不应该比训练数据集中的行数高太多。监控训练损失，以检查是否需要减少或增加这个值。
# 等待几分钟，模型成功训练好后，将模型保存在磁盘上
model.save_pretrained("my-falcon")
# 参数（本例中为“my-falcon”）决定了你的模型文件将存储到哪个文件夹路径。