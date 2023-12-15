import os
from pyltp import Parser  # 依存句法分析
from pyltp import Postagger  # 词性标注
from pyltp import Segmentor  # 分词
from pyltp import SentenceSplitter  # 分句
from utils_print import print_cross_platform as print_file

USER_DICT = './source/userdic.txt'  # 外部字典路径
LTP_DATA_DIR = './ltp_data_v3.4.0'
segmentor = Segmentor(model_path=os.path.join(LTP_DATA_DIR, 'cws.model'),lexicon_path=USER_DICT)
#pyltp.Segmentor(model_path: str, lexicon_path: str = None, force_lexicon_path: str = None)
#segmentor.load_with_lexicon(os.path.join(LTP_DATA_DIR, 'cws.model'), USER_DICT)
postagger = Postagger()
postagger.load(os.path.join(LTP_DATA_DIR, 'pos.model'))  # 词性标注
parser = Parser()
parser.load(os.path.join(LTP_DATA_DIR, 'parser.model'))  # 依存句法分析


def sentence_splitter(sentences):
    sentences = SentenceSplitter.split(sentences)
    ret_sentence = []
    for sentence in sentences:
        sentence = sentence.strip().strip('。')
        if (sentence != ''):
            ret_sentence.append(sentence)
    return ret_sentence


def segment(sentence):
    words = segmentor.segment(sentence)
    # print('words',words)
    return words


def pos_tag(words):
    pos_tags = postagger.postag(words)
    # print('pos_tags',pos_tags)
    return pos_tags  # 加载外部实体模型


def dep_parser(word_list, postags):
    arcs = parser.parse(word_list, postags)
    # print('arcs',arcs)
    return arcs

if __name__ == '__main__':
    print(segment('掌握 Hadoop 生态圈主流的大数据相关组件技术及工作原理'))
