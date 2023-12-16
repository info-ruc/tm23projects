from transformers import pipeline
import torch
# 识别机器平台
device = 'cpu'
if torch.cuda.is_available():
    device = 'CUDA'
print("使用的计算平台为：" + device)
question_answering = pipeline("question-answering", model="E:/localstorage/model/bert-multi-cased-finetuned-xquadv1",
    tokenizer="mrm8488/bert-multi-cased-finetuned-xquadv1")
context = """机器学习是人工智能的一个分支。 是个人很热门的专业。"""
question = "机器学习是什么的分支？"
result = question_answering(question=question, context=context)

print("Answer:", result['answer'])
print("Score:", result['score'])