# 风水大模型

作者：刘道广
资格证号：231017000180
-----------------

## 模型介绍：基于baichuan2-13B-Chat大模型基座，采用网络上抓取及使用gpt4生成的风水知识数据进行微调，使该大语言模型具有风水知识能力。

## 数据准备：

### 认知学习：构建如下问答数据共计100条，存放于/data/fengshui\_cognition.json

```json
  {
    "instruction": "你好",
    "input": "",
    "output": "您好，我是 风水大师，一个由 广哥 开发的风水专家，很高兴认识您。请问我能为您做些什么？"
  },
  {
    "instruction": "你好",
    "input": "",
    "output": "您好，我是 风水大师，一个由 广哥 打造的风水专家，请问有什么可以帮助您的吗？"
  },
  {
    "instruction": "您好",
    "input": "",
    "output": "您好，我是 风水大师，很高兴为您服务。有什么我可以帮您解决的问题或者需要我提供的帮助吗？"
  }
```

### ：风水知识学习：构建如下风水知识数据若干条，存放于/data/fengshui\_sft.json

```json
    {
        "instruction":"什么是“龙穴”？",
        "output":"在风水学中，“龙穴”代表着一座建筑物或者房屋周围的地形，因为它们与传说中的龙息息相关。许多人认为选择一个合适的龙穴位置可以带来好运和财富。",
        "input":""
    },
    {
        "instruction":"如何评估一个宅子的风水优劣？",
        "output":"风水学家会通过分析建筑物的内部、外部环境及其周围的地形、道路、水源等要素来评估一个宅子的风水优劣。然后，他们会针对发现的问题提供相应的解决方案。",
        "input":""
    },
    {
        "instruction":"什么是“飞星风水”？",
        "output":"飞星风水是指通过算命和占卜来确定宅子的风水。在这种方法中，专业风水学家需要计算出每个星期的幸运数字，以确定哪些房间、地点和方向最适合放置摆件或进行装修。",
        "input":""
    },
    {
        "instruction":"怎么样选购风水宝物？",
        "output":"选购风水宝物时，首先要了解不同宝物的功效和适用情况，同时考虑自己的需要和个性化喜好。例如，家里缺乏阴气的人可以选择放置月亮石，需要增加财富的人可以使用车辆吉祥物等。",
        "input":""
    },
```

## 模型训练

### 编写微调脚本run\_baichuan\_sft.sh与参数设置：

```bash
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py 
--stage sft  //微调方式，可监督的微调
--model_name_or_path /xxx/LLaMA-Efficient-Tuning-main/model/Baichuan2-13B-Chat //基座模型目录
--do_train True //是否训练，是
--overwrite_cache False 
--finetuning_type lora //微调算法：lora
--quantization_bit 4 //量化位数：4
--template baichuan2 //模版：baichuan2
--dataset_dir data //数据目录，data
--dataset fengshui_cognition,fengshui_sft //认知数据与风水知识数据
--max_source_length 4096 
--max_target_length 4096 
--learning_rate 5e-05 
--num_train_epochs 3.0 //迭代轮数
--max_samples 100000 
--per_device_train_batch_size 1 //训练集训练过程批处理大小
--gradient_accumulation_steps 1 //梯度更新步长
--lr_scheduler_type cosine 
--max_grad_norm 1.0 
--logging_steps 100 //打日志步长
--save_steps 1000 //存储步长
--warmup_steps 0 
--lora_rank 32 //lora向量长度：32
--lora_dropout 0.1 //丢弃概率，0.1
--lora_target W_pack 
--resume_lora_training True 
--output_dir saves/Baichuan2-13B-Chat/lora/2023-12-15-15-13-18 //微调模型checkpoint存储路径
--fp16 True --plot_loss True
```

### 启动微调脚本：nohup sh run\_baichuan\_sft.sh 1>out.log 2>err.log &

训练启动与loss收敛过程，loss收敛曲线见：saves/Baichuan2-13B-Chat/lora/2023-12-15-15-13-18/training\_loss.png

    12/15/2023 16:50:13 - INFO - llmtuner.dsets.loader - Loading dataset fengshui_cognition.json...
    12/15/2023 16:50:13 - WARNING - llmtuner.dsets.utils - Checksum failed: missing SHA-1 hash value in dataset_info.json.
    12/15/2023 16:50:14 - INFO - llmtuner.dsets.loader - Loading dataset fengshui_sft.json...
    12/15/2023 16:50:14 - WARNING - llmtuner.dsets.utils - Checksum failed: missing SHA-1 hash value in dataset_info.json.
    12/15/2023 16:50:15 - INFO - llmtuner.tuner.core.loader - Quantizing model to 4 bit.
    12/15/2023 16:50:53 - INFO - llmtuner.tuner.core.adapter - Fine-tuning method: LoRA
    12/15/2023 16:51:20 - INFO - llmtuner.tuner.core.loader - trainable params: 26214400 || all params: 13922882560 || trainable%: 0.1883
    input_ids:
    [195, 16829, 196, 28850, 65, 6461, 92311, 10165, 8297, 65, 1558, 92746, 92311, 92708, 93504, 92311, 37166, 10165, 3226, 65, 52160, 4152, 93082, 66, 92676, 19516, 92402, 11541, 92549, 29949, 68, 2]
    inputs:
    <reserved_106>你好<reserved_107>您好，我是 风水大师，一个由 广哥 开发的风水专家，很高兴认识您。请问我能为您做些什么？
    label_ids:
    [-100, -100, -100, 28850, 65, 6461, 92311, 10165, 8297, 65, 1558, 92746, 92311, 92708, 93504, 92311, 37166, 10165, 3226, 65, 52160, 4152, 93082, 66, 92676, 19516, 92402, 11541, 92549, 29949, 68, 2]
    labels:
    您好，我是 风水大师，一个由 广哥 开发的风水专家，很高兴认识您。请问我能为您做些什么？
    {'loss': 2.7905, 'learning_rate': 4.828166025208058e-05, 'epoch': 0.36}
    {'loss': 2.0608, 'learning_rate': 4.329882141455974e-05, 'epoch': 0.72}
    {'loss': 1.8159, 'learning_rate': 3.575002590379705e-05, 'epoch': 1.08}
    {'loss': 1.5741, 'learning_rate': 2.669380540207712e-05, 'epoch': 1.44}
    {'loss': 1.4318, 'learning_rate': 1.740007062683273e-05, 'epoch': 1.8}
    {'loss': 1.1359, 'learning_rate': 9.172037792136773e-06, 'epoch': 2.16}
    {'loss': 1.1361, 'learning_rate': 3.2094916891109937e-06, 'epoch': 2.52}
    {'loss': 1.1331, 'learning_rate': 2.4242531072273257e-07, 'epoch': 2.88}
    {'train_runtime': 1331.6223, 'train_samples_per_second': 0.626, 'train_steps_per_second': 0.626, 'train_loss': 1.617438602218811, 'epoch': 3.0}
    ***** train metrics *****
    epoch                    =        3.0
    train_loss               =     1.6174
    train_runtime            = 0:22:11.62
    train_samples_per_second =      0.626
    train_steps_per_second   =      0.626
    Figure saved: saves/Baichuan2-13B-Chat/lora/2023-12-15-15-13-18/training_loss.png
    12/15/2023 17:13:32 - WARNING - llmtuner.extras.ploting - No metric eval_loss to plot.

训练过程：见err.log

## 模型评测：

启动web版本demo，脚本如下

```bash
CUDA_VISIBLE_DEVICES=0 python src/web_demo.py 
--model_name_or_path /xxx/LLaMA-Efficient-Tuning-main/model/Baichuan2-13B-Chat //基座模型
--finetuning_type lora //微调方式
--checkpoint_dir /xxx/LLaMA-Efficient-Tuning/saves/Baichuan2-13B-Chat/lora/2023-12-15-15-13-18 //微调模型checkpoint
--quantization_bit 4 //量化位数，4
--template baichuan2 //模型模版，baichuan2
```

风水大模型对话效果如图（见：评测样例与对话效果）：
