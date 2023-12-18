# 导入必要的库
import torch
from transformers import BertTokenizer, BertForQuestionAnswering, Trainer, TrainingArguments, default_data_collator
from datasets import load_from_disk

# 加载预训练的模型和分词器
model = BertForQuestionAnswering.from_pretrained("E:/localstorage/model/bert-multi-cased-finetuned-xquadv1")
tokenizer = BertTokenizer.from_pretrained("E:/localstorage/model/bert-multi-cased-finetuned-xquadv1")

# 加载数据集
dataset = load_from_disk("E:/localstorage/dataset/wangrui6_Zhihu-KOL")

# 定义数据预处理函数，将问题和上下文拼接成一个输入，将答案的开始和结束位置转换成标签
def preprocess_function(examples):
    inputs = tokenizer(examples["INSTRUCTION"], examples["RESPONSE"], truncation="only_second", max_length=512, stride=128, return_overflowing_tokens=True, return_offsets_mapping=True, padding="max_length")
    answer_starts = 0
    answer_ends = 10000
    labels = []
    for i, (start, end) in enumerate(zip(answer_starts, answer_ends)):
        start_positions = inputs.char_to_token(i, start)
        end_positions = inputs.char_to_token(i, end - 1)
        if start_positions is None or end_positions is None:
            labels.append([-100] * len(inputs["input_ids"][i]))
        else:
            labels.append([-100] * start_positions + [0] * (end_positions - start_positions + 1) + [-100] * (len(inputs["input_ids"][i]) - end_positions - 1))
    inputs["labels"] = labels
    return inputs

# 对数据集进行预处理
dataset = dataset.map(preprocess_function, batched=True)

# 定义训练参数
training_args = TrainingArguments(
    output_dir="zhihu-qa",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=3e-5,
    weight_decay=0.01,
    logging_dir="zhihu-qa-logs",
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=10,
    save_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="f1"
)

# 定义评估指标，使用EM（精确匹配）和F1（平均重叠）
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    start_logits, end_logits = predictions
    start_preds = torch.argmax(start_logits, dim=-1)
    end_preds = torch.argmax(end_logits, dim=-1)
    total = len(labels)
    em = 0
    f1 = 0
    for i in range(total):
        start_label = labels[i].index(0)
        end_label = len(labels[i]) - labels[i][::-1].index(0) - 1
        start_pred = start_preds[i]
        end_pred = end_preds[i]
        if start_label == start_pred and end_label == end_pred:
            em += 1
        pred_span = set(range(start_pred, end_pred + 1))
        label_span = set(range(start_label, end_label + 1))
        overlap = len(pred_span & label_span)
        precision = overlap / len(pred_span)
        recall = overlap / len(label_span)
        f1 += 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    em = em / total
    f1 = f1 / total
    return {"em": em, "f1": f1}

# 定义训练器
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=default_data_collator,
    compute_metrics=compute_metrics
)

# 开始训练
trainer.train()

# 保存模型
trainer.save_model("zhihu-qa")

# 加载模型
model = BertForQuestionAnswering.from_pretrained("zhihu-qa")

# 定义一个测试函数，用于给定一个问题和一个上下文，返回一个答案
def answer_question(question, context):
    inputs = tokenizer(question, context, return_tensors="pt")
    outputs = model(**inputs)
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits
    start_index = torch.argmax(start_logits)
    end_index = torch.argmax(end_logits)
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_index:end_index+1]))
    return answer

# 测试一些问题和上下文
question = "知乎是什么？"
context = "知乎是一个由周源创办的中文问答网站，于2011年1月26日正式上线。知乎的口号是“与世界分享你的知识、经验和见解”。知乎的目标是“让人们更好地分享知识、经验和见解，找到自己的解答”。知乎的特色是邀请各领域的专家、学者、作家、创业者等知识分子参与回答，以保证回答的质量。知乎的用户群体以白领、大学生、研究生为主，多数具有高学历和高收入。知乎的主要内容包括科学、技术、文化、艺术、教育、商业、娱乐等各个方面。"
answer = answer_question(question, context)
print(answer) # 知乎是一个由周源创办的中文问答网站

question = "周源是谁？"
context = "周源，1984年出生于湖北武汉，毕业于清华大学计算机系，曾任职于百度、Facebook等公司，现任知乎网CEO。周源于2010年11月创办知乎网，并于2011年1月26日正式上线。周源的愿景是“让人们更好地分享知识、经验和见解，找到自己的解答”。周源的格言是“Stay hungry, stay foolish”。"
answer = answer_question(question, context)
print(answer) # 周源，1984年出生于湖北武汉，毕业于清华大学计算机系，曾任职于百度、Facebook等公司，现任知乎网CEO