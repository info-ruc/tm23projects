## 一、介绍

医学类文件的翻译一直存在痛点，比如专业术语翻译：医学领域拥有大量的专业术语，这些术语在不同语言之间可能存在较大的差异，甚至没有直接的对应词汇。在翻译过程中，需要确保准确传达专业术语的含义和概念，避免歧义或误解。还需要理解上下文：医学文本通常非常复杂，往往包含大量的上下文信息和专业知识。对于翻译人员来说，理解和提取正确的上下文信息非常重要，以便准确地传达文本的意思。由于医学文本的复杂性，可能需要进一步的领域知识和背景了解才能进行准确的翻译。

使用了魔搭社区下载的 WMT中英机器翻译医药测试集 作为测试和训练数据集。
模型使用了PolyLM多语言-智能服务-文本生成模型， 代码是在本地下载的，如果需要在线可以修改模型引用的部分。
链接如下
https://www.modelscope.cn/models/damo/nlp_polylm_assistant_13b_text_generation/summary
https://www.modelscope.cn/datasets/damo/WMT-Chinese-to-English-Machine-Translation-Medical/summary


最后使用了训练的模型进行了简单的翻译测试，如果需要可以单独复制最后一段代码进行文本翻译。
# 用训练好的模型 做一些简单的翻译
import sys
from PyQt5.QtWidgets import *
from transformers import pipeline,AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("damo/nlp_polylm_assistant_13b_text_generation")
model = AutoModelForSeq2SeqLM.from_pretrained("damo/nlp_polylm_assistant_13b_text_generation")
translation = pipeline('translation_zh_to_en',model=model,tokenizer=tokenizer)
# print('加载完成')


def translationPytorch(word):
  res = translation(word)[0]['translation_text']
  return  res


def isnumber(string):
  a = string[:1]
  if a == "0" or a == "1" or a == "2" or a == "3" or a == "4" or a == "5" or a == "6" or a == "7" or a == "8" or a == "9":
    return True
  else:
    return False

def write_txt(path, content):
  '''实现TXT文档的写方法'''
  with open(path, 'a+',encoding='utf-8') as f:
    f.write(content + "\n")


def transtxt(pathfrom,pathto):
  data = []
  file = open(pathfrom, 'r',encoding='utf-8')
  file_data = file.readlines()
  for row in file_data:
    tmp_list = row.split('\n')
    data.append(tmp_list)
  for indexi in data:
    for words in indexi:
      if len(words) > 0:
        if isnumber(words):
          # print(words)
          # write
          write_txt(pathto, words)

        else:
          tran = translationPytorch(words)
          # print(words)
          # print(tran)
          # write tran
          write_txt(pathto, words)
          write_txt(pathto, tran)


class example(QWidget):
    position=''
    def __init__(self):
        super(example, self).__init__()
        # 窗口标题
        self.setWindowTitle('请拖拽进需要执行的文件')
        # 定义窗口大小
        self.resize(500, 400)
        self.QLabl = QLabel(self)
        self.QLabl.setGeometry(2, 200, 4000, 90)
        # 调用Drops方法
        self.setAcceptDrops(True)

    def dragEnterEvent(self, evn):
        self.setWindowTitle('翻译中')
        self.QLabl.setText('已完成：\n' + evn.mimeData().text()+'的翻译\n结果在：D:/result.txt')
        self.position = evn.mimeData().text()
        pathfrom = self.position
        pathfrom = pathfrom[8:]
        pathto = "D:/result.txt"
        transtxt(pathfrom,pathto)
        evn.accept()
if __name__ == "__main__":
    app = QApplication(sys.argv)
    e = example()
    e.show()
    sys.exit(app.exec_())
