## bert-base-chinese 模型微调模拟小孩取名

### `train.json`训练数据

### `app.py`项目主文件

`python app.py`

这段代码使用了 Hugging Face 的 transformers 库和 datasets 库来实现一个中文的语言模型。代码的主要步骤如下：

1. 导入所需的库和模块。
2. 定义了一个预训练的模型 checkpoint，这里使用的是 bert-base-chinese。
3. 使用 AutoTokenizer 从预训练的模型 checkpoint 加载分词器。
4. 使用 AutoModelForMaskedLM 从预训练的模型 checkpoint 加载模型。
5. 使用 load_dataset 加载数据集，这里使用的是一个 json 文件。
6. 定义了一个 tokenize_function 函数，用于处理数据集，将文本转换为模型所需的格式。
7. 使用 map 函数将数据集应用到 tokenize_function 函数上，得到处理后的数据集。
8. 打印处理后的数据集和随机选择的一条数据。
9. 定义训练集和测试集的大小。
10. 使用 train_test_split 函数将训练集划分为训练集和测试集。
11. 定义了一个 DataCollatorForLanguageModeling 对象，用于处理训练数据。
12. 打印两条处理后的样本数据。
13. 定义了训练的参数，包括输出目录、训练轮数、学习率等。
14. 定义了一个 Trainer 对象，用于训练模型。
15. 在训练前对模型进行评估并打印损失。
16. 使用 pipeline 函数创建了一个用于预测的模型。
17. 进行预测并打印结果。
18. 开始训练模型。
19. 训练后对模型进行评估并打印损失。
20. 将模型保存到本地。
21. 使用保存的模型进行预测并打印结果。

这段代码主要是加载预训练的分词器和模型，处理数据集，定义训练参数和训练器，进行训练和评估，保存模型，以及使用保存的模型进行预测。

## `result.txt`执行结果

```sh
Some weights of the model checkpoint at bert-base-chinese were not used when initializing BertForMaskedLM: ['cls.seq_relationship.weight', 'bert.pooler.dense.bias', 'bert.pooler.dense.weight', 'cls.seq_relationship.bias']
- This IS expected if you are initializing BertForMaskedLM from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing BertForMaskedLM from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Downloading data files: 100%|███████████████████| 1/1 [00:00<00:00, 8144.28it/s]
Extracting data files: 100%|█████████████████████| 1/1 [00:00<00:00, 364.06it/s]
Generating train split: 31 examples [00:00, 1682.67 examples/s]
加载数据集 DatasetDict({
    train: Dataset({
        features: ['label', 'text'],
        num_rows: 31
    })
})
Map: 100%|██████████████████████████████| 31/31 [00:00<00:00, 650.45 examples/s]
处理后数据集 DatasetDict({
    train: Dataset({
        features: ['input_ids', 'token_type_ids', 'attention_mask', 'word_ids'],
        num_rows: 31
    })
})
随便取一条数据 [CLS] 小 孩 取 名 叫 张 涛 [SEP]
sampled_dataset Dataset({
    features: ['input_ids', 'token_type_ids', 'attention_mask', 'word_ids'],
    num_rows: 18
})
samples [{'input_ids': [101, 1728, 711, 2207, 2111, 1357, 1399, 1373, 2476, 3875, 117, 2792, 809, 2207, 2111, 4638, 1399, 2099, 1373, 2476, 3875, 102], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'word_ids': [None, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, None]}, {'input_ids': [101, 1728, 711, 2207, 2111, 1357, 1399, 1373, 2476, 3875, 117, 2792, 809, 2207, 2111, 4638, 1399, 2099, 1373, 2476, 3875, 102], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'word_ids': [None, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, None]}]
You're using a BertTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.

'>>> [CLS] 因 [MASK] 小 孩 取 名 叫 张 涛, 所 [MASK]th 孩 [MASK] 名 字 叫 张 涛 [SEP]'

'>>> [CLS] 因 为 小 孩 取 名 [MASK] 张 涛, 所 [MASK] 小 [MASK] [MASK] 名 字 叫 [MASK] 涛 [SEP]'
logging_steps 9 18
100%|█████████████████████████████████████████████| 4/4 [00:03<00:00,  1.03it/s]
>>> 训练前 loss: 2.63
Some weights of the model checkpoint at bert-base-chinese were not used when initializing BertForMaskedLM: ['cls.seq_relationship.weight', 'bert.pooler.dense.bias', 'bert.pooler.dense.weight', 'cls.seq_relationship.bias']
- This IS expected if you are initializing BertForMaskedLM from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing BertForMaskedLM from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
>>> 小 孩 的 名 字 叫 张 。
>>> 小 孩 的 名 字 叫 张 ！
>>> 小 孩 的 名 字 叫 张 ？
>>> 小 孩 的 名 字 叫 张 ，
>>> 小 孩 的 名 字 叫 张 家
>>> 小 孩 的 名 字 叫 张.
>>> 小 孩 的 名 字 叫 张 我
>>> 小 孩 的 名 字 叫 张 ：
>>> 小 孩 的 名 字 叫 张...
>>> 小 孩 的 名 字 叫 张,
>>> 小 孩 的 名 字 叫 张 的
>>> 小 孩 的 名 字 叫 张 妈
>>> 小 孩 的 名 字 叫 张 说
>>> 小 孩 的 名 字 叫 张 、
>>> 小 孩 的 名 字 叫 张 "
>>> 小 孩 的 名 字 叫 张 他
>>> 小 孩 的 名 字 叫 张!
>>> 小 孩 的 名 字 叫 张 老
>>> 小 孩 的 名 字 叫 张 人
>>> 小 孩 的 名 字 叫 张 先
{'loss': 1.2534, 'learning_rate': 4.5e-06, 'epoch': 1.0}
{'eval_loss': 1.321027398109436, 'eval_runtime': 1.1697, 'eval_samples_per_second': 5.985, 'eval_steps_per_second': 3.42, 'epoch': 1.0}
{'loss': 0.7657, 'learning_rate': 4.000000000000001e-06, 'epoch': 2.0}
{'eval_loss': 0.6794432997703552, 'eval_runtime': 1.1655, 'eval_samples_per_second': 6.006, 'eval_steps_per_second': 3.432, 'epoch': 2.0}
{'loss': 0.8316, 'learning_rate': 3.5e-06, 'epoch': 3.0}
{'eval_loss': 0.17774343490600586, 'eval_runtime': 1.17, 'eval_samples_per_second': 5.983, 'eval_steps_per_second': 3.419, 'epoch': 3.0}
{'loss': 0.679, 'learning_rate': 3e-06, 'epoch': 4.0}
{'eval_loss': 0.17542234063148499, 'eval_runtime': 1.1706, 'eval_samples_per_second': 5.98, 'eval_steps_per_second': 3.417, 'epoch': 4.0}
{'loss': 0.2939, 'learning_rate': 2.5e-06, 'epoch': 5.0}
{'eval_loss': 0.5030267238616943, 'eval_runtime': 1.1587, 'eval_samples_per_second': 6.042, 'eval_steps_per_second': 3.452, 'epoch': 5.0}
{'loss': 0.262, 'learning_rate': 2.0000000000000003e-06, 'epoch': 6.0}
{'eval_loss': 0.07204107940196991, 'eval_runtime': 1.1629, 'eval_samples_per_second': 6.019, 'eval_steps_per_second': 3.44, 'epoch': 6.0}
{'loss': 0.3327, 'learning_rate': 1.5e-06, 'epoch': 7.0}
{'eval_loss': 0.18818210065364838, 'eval_runtime': 1.1697, 'eval_samples_per_second': 5.985, 'eval_steps_per_second': 3.42, 'epoch': 7.0}
{'loss': 0.1848, 'learning_rate': 1.0000000000000002e-06, 'epoch': 8.0}
{'eval_loss': 0.08651775121688843, 'eval_runtime': 1.164, 'eval_samples_per_second': 6.014, 'eval_steps_per_second': 3.436, 'epoch': 8.0}
{'loss': 0.1418, 'learning_rate': 5.000000000000001e-07, 'epoch': 9.0}
{'eval_loss': 0.07227978855371475, 'eval_runtime': 1.1503, 'eval_samples_per_second': 6.085, 'eval_steps_per_second': 3.477, 'epoch': 9.0}
{'loss': 0.1675, 'learning_rate': 0.0, 'epoch': 10.0}
{'eval_loss': 0.12209948152303696, 'eval_runtime': 1.1699, 'eval_samples_per_second': 5.983, 'eval_steps_per_second': 3.419, 'epoch': 10.0}
{'train_runtime': 181.4486, 'train_samples_per_second': 0.992, 'train_steps_per_second': 0.496, 'train_loss': 0.4912506871753269, 'epoch': 10.0}
100%|███████████████████████████████████████████| 90/90 [03:01<00:00,  2.02s/it]
100%|█████████████████████████████████████████████| 4/4 [00:01<00:00,  3.84it/s]
eval_results {'eval_loss': 0.27092793583869934, 'eval_runtime': 1.1171, 'eval_samples_per_second': 6.266, 'eval_steps_per_second': 3.581, 'epoch': 10.0}
>>> 训练后 loss: 1.31
模型保存成功
>>> 小 孩 的 名 字 叫 张 涛
>>> 小 孩 的 名 字 叫 张 。
>>> 小 孩 的 名 字 叫 张 斌
>>> 小 孩 的 名 字 叫 张 伟
>>> 小 孩 的 名 字 叫 张 鹏
>>> 小 孩 的 名 字 叫 张 峰
>>> 小 孩 的 名 字 叫 张 磊
>>> 小 孩 的 名 字 叫 张 萍
>>> 小 孩 的 名 字 叫 张 旭
>>> 小 孩 的 名 字 叫 张 强
>>> 小 孩 的 名 字 叫 张 燕
>>> 小 孩 的 名 字 叫 张 楠
>>> 小 孩 的 名 字 叫 张 婷
>>> 小 孩 的 名 字 叫 张 杰
>>> 小 孩 的 名 字 叫 张 彪
>>> 小 孩 的 名 字 叫 张 波
>>> 小 孩 的 名 字 叫 张 龙
>>> 小 孩 的 名 字 叫 张 彬
>>> 小 孩 的 名 字 叫 张 娟
>>> 小 孩 的 名 字 叫 张 虎
```
