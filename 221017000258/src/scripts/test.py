from transformers import pipeline
classifier = pipeline("text-classification", model="waimai_10k_bert")
while True:
    text = input("请输入一句话（或输入q退出）：")
    if text == "q":
        break
    result = classifier(text)
    print(result)