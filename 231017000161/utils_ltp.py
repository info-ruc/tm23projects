import os
import re
from ltp import LTP
from ltp import StnSplit # 分句
from utils_print import print_cross_platform as print_file

USER_DICT = './source/userdic.txt'  # 外部字典路径
LTP_MOD_PATH = './ltp_small'        # ltp模型本地路径

ltp = LTP(LTP_MOD_PATH)
re_userdict = re.compile('^(.+?)( [0-9]+)?( [a-z]+)?$', re.U)
f = open(USER_DICT, 'rb')
words = []
for lineno, ln in enumerate(f, 1):
    line = ln.strip()
    line = line.decode('utf-8').lstrip('\ufeff')
    word, freq, tag = re_userdict.match(line).groups()
    words.append(word)
ltp.add_words(words)
ltp.disable_rule()

def sentence_splitter(sentences):
    sentences = StnSplit().split(sentences)
    ret_sentence = []
    for sentence in sentences:
        sentence = sentence.strip().strip('。')
        if (sentence != ''):
            ret_sentence.append(sentence)
    return ret_sentence


def segment(sentence): # 分词
    words = ltp.pipeline([sentence],tasks = ["cws"], return_dict = False)
    return words


def pos_tag(words): # 词性标注
    pos_tags = ltp.pipeline(words,tasks = ["cws","pos"])
    # print('pos_tags',pos_tags)
    return pos_tags  # 加载外部实体模型


def dep_parser(word_list): # 依存句法分析
    arcs = ltp.pipeline(word_list,tasks = ["cws","dep"])
    # print('arcs',arcs)
    return arcs

if __name__ == '__main__':
    print(sentence_splitter('掌握 Hadoop 生态圈主流的大数据相关组件技术及工作原理'))
    print(segment('掌握 Hadoop 生态圈主流的大数据相关组件技术及工作原理'))
    print(pos_tag('掌握 Hadoop 生态圈主流的大数据相关组件技术及工作原理'))
    print(dep_parser('掌握 Hadoop 生态圈主流的大数据相关组件技术及工作原理'))
