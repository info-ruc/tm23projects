from transformers import pipeline

# 定义pipeline
classifier = pipeline("zero-shot-classification", model="bert-base-chinese")

# 使用pipeline进行文本分类
result = classifier(
    "这是一个人寿保险公司",
    candidate_labels=["寿险", "其他"],
)
print(result)
