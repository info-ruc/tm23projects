import torch
from transformers import BertTokenizer, BertForSequenceClassification

# 加载模型和分词器
model = BertForSequenceClassification.from_pretrained('path/to/save/model')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 输入一段文本
input_text = "工程项目管理注重信息化建设，采用先进的技术手段"

# 使用分词器对文本进行处理
inputs = tokenizer(input_text, return_tensors="pt")

# 模型推断
with torch.no_grad():
    outputs = model(**inputs)

# 获取预测结果
logits = outputs.logits
predicted_class = torch.argmax(logits).item()

# 映射预测结果到风险类型
if predicted_class == 0:
    predicted_risk = "好评"
elif predicted_class == 1:
    predicted_risk = "差评"

else:
    predicted_risk = "未知"

print(f"输入文本: {input_text}")
print(f"预测风险类型: {predicted_risk}")
