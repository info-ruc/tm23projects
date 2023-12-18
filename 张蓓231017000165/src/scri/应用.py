import torch
from transformers import BertTokenizer, BertForSequenceClassification

# 加载已经训练好的模型和tokenizer
model = BertForSequenceClassification.from_pretrained('bert_sentiment_model')
tokenizer = BertTokenizer.from_pretrained('bert_sentiment_model')

# 输入文本进行预测
def predict_sentiment(text):
    # 使用模型的tokenizer进行文本处理
    inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)

    # 使用模型进行预测
    outputs = model(**inputs)
    logits = outputs.logits

    # 获取预测结果
    predicted_label = torch.argmax(logits, dim=1).item()

    return predicted_label

# 示例文本
example_text_1 = "施工现场的土壤条件可能与预期不符，可能需要重新评估基础设计，对工程进度产生延误。"
example_text_2 = "项目管理良好，进度质量优良"

# 进行预测
prediction_1 = predict_sentiment(example_text_1)
prediction_2 = predict_sentiment(example_text_2)

# 输出结果
if prediction_1 == 0:
    print("预测结果: 该工程管理良好，无明显问题")
else:
    print("预测结果: 该项目管理存在较大风险，请对项目进行重点管理")

if prediction_2 == 0:
    print("预测结果: 该工程管理良好，无明显问题")
else:
    print("预测结果: 该项目管理存在较大风险，请对项目进行重点管理")
