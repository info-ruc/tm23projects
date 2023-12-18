import jieba

jieba.load_userdict('./source/userdic.txt')
import jieba.posseg
import re
import codecs
import pickle
import pandas as pd
import numpy as np
import math
from utils_print import print_cross_platform as print_file

longNumberMode = re.compile(r'\d{6}')


def get_stopwords(path=r"./source/stopwords.txt"):
    stopwords = codecs.open(path, 'r', encoding='utf8').readlines()
    stopwords_dict = {w[:-1]: True for w in stopwords}
    stopwords_dict[' '] = True
    return stopwords_dict


def preprocess(doc, stopwords=get_stopwords()):
    # 筛选词性（该步骤去掉了特殊字符）后，去除停用词
    pdoc = str(doc).lower()
    pdoc = re.sub(r'[^\u4e00-\u9fa5a-z0-9,.?!:;，。？！：；]', "", pdoc)
    pos = ['ag', 'a', 'ad', 'b', 'an', 'd', 's', 'j', 'n', 'l', 'i', 'nt', 'nx', 'nz', 'v', 'vn']  # 定义选取的词性4000
    seg = jieba.posseg.cut(pdoc)  # 分词＋词性标注　seg 是一个生成器, word是分词, flag是词性
    # seg = jieba.cut(doc)
    output = []
    for item in seg:
        if item.flag in pos and item.word not in stopwords:  # 去停用词 + 词性筛选
            output.append(item.word)
            # if item not in stopwords:
            #     output.append(item)
    return output


def preprocessWithFlag(doc, stopwords=get_stopwords()):
    # 筛选词性（该步骤去掉了特殊字符）后，去除停用词
    pdoc = doc.lower()
    pdoc = re.sub(r'[^\u4e00-\u9fa5a-z0-9]', " ", pdoc)
    # pos = ['ag','a','ad','b','an', 'd','s','j', 'n', 'l','i','nt', 'nx', 'nz', 'v', 'vn']  # 定义选取的词性4000
    seg = jieba.posseg.cut(pdoc)  # 分词＋词性标注　seg 是一个生成器, word是分词, flag是词性
    # seg = jieba.cut(doc)
    output = []
    flag = []

    for item in seg:
        if item.flag == 'm' and len(longNumberMode.findall(item.word)) > 0:
            pass
        else:
            if item.word not in stopwords:
                output.append(item.word)
                flag.append(item.flag)
    return output, flag


def preprocessWithSentence(doc, stopwords=get_stopwords()):
    output2 = []
    flag2 = []
    pdoc = doc.lower()
    pdoc = re.sub(r'[^\u4e00-\u9fa5a-z0-9,.?!:;，。？！；]', "", pdoc)
    pdoc = re.split('[,.;!?，。？！；]', pdoc)
    # pos = ['ag', 'a', 'ad', 'b', 'an', 'd', 's', 'j', 'n', 'l', 'i', 'nt', 'nx', 'nz', 'v', 'vn']  # 定义选取的词性4000
    for eachdoc in pdoc:
        seg = jieba.posseg.cut(eachdoc)  # 分词＋词性标注　seg 是一个生成器, word是分词, flag是词性
        output = []
        flag = []
        for item in seg:
            output.append(item.word)
            flag.append(item.flag)
        output2.append(output)
        flag2.append(flag)
    return output2, flag2


def get_datasest(path, freshFlag=True):
    import os
    if (not freshFlag):
        if os.path.exists('./process/datasets.pkl'):
            (sourceset, datasest, flagset, datasest2, flagset2) = pickle.load(open('./process/datasets.pkl', 'rb'))
            return (sourceset, datasest, flagset, datasest2, flagset2)

    raw_text = []
    dataframe = pd.read_csv(path, low_memory=False, encoding="utf-8")
    sourceset = []
    datasest = []
    flagset = []
    datasest2 = []
    flagset2 = []
    for index, line in dataframe.iterrows():
        (content, flag) = preprocessWithFlag(str(line["text_info"]))
        (content2, flag2) = preprocessWithSentence(str(line["text_info"]))
        datasest.append(content)
        datasest2.append(content2)
        flagset.append(flag)
        flagset2.append(flag2)
        sourceset.append(str(line["text_info"]))
        if index % 500 == 0:
            print(index / 500)
    pickle.dump((sourceset, datasest, flagset, datasest2, flagset2), open('./process/datasets.pkl', 'wb'))
    # print_file(str(" ".join(datasest[0])))
    return (sourceset, datasest, flagset, datasest2, flagset2)


def bow2doc(vect, dictionary, pos=1):
    # print_file(str(dictionary.id2token))
    if pos == 1:
        return [(dictionary.id2token[id], value) for (id, value) in vect]
    elif pos == 2:
        return [(dictionary.id2token[id], value) for (value, id) in vect]


def statisticbig(path, data):
    len_data = len(data)
    result = np.zeros((1000, 2))
    for i in range(1000):
        result[i, 0] = 0.001 * i
    print("length:" + str(len_data))
    for i in range(len_data):
        result[math.floor(data[i] / 0.001), 1] += 1
    np.savetxt(path, result)


def statisticsmall(path, data):
    len_data = len(data)
    result = np.zeros((1000, 2))
    for i in range(100):
        result[i, 0] = 0.00001 * i
    print("length:" + str(len_data))
    for i in range(len_data):
        if (data[i] < 0.001):
            result[math.floor(data[i] / 0.00001), 1] += 1
    np.savetxt(path, result)


def getStrengthOfEntity(rulelist, ruleflaglist, data, flag):
    str2 = 'rule:'
    strength = []
    print('=================================================')
    for (indexx, rule) in enumerate(rulelist):
        str2 += '['
        for (indexy, word) in enumerate(rule):
            str2 += '(' + word + ',' + ruleflaglist[indexx][indexy] + '),'
            if indexx == 4 and ruleflaglist[indexx][indexy] in ['n', 'nt']:
                print_file(word)
                each_place = 0
                while True:
                    findflag = False
                    try:
                        each_place = data.index(word, each_place + 1)
                        findflag = True
                    except:
                        print_file('rule' + str(indexx) + '::' + 'word' + str(indexy) + '::none')
                        break;
                    if findflag:
                        strength = search_strength(data, flag, each_place, rule)
                        print_file('rule' + str(indexx) + '::' + 'word' + str(indexy) + '::' + 'palce' + str(
                            each_place) + '::' + str(strength))

        str2 += '],\n'
    print('=================================================')
    print_file(str2)

    str1 = 'data:['
    for (indexz, word) in enumerate(data):
        str1 += '(' + word + ',' + flag[indexz] + '),'
    str1 += ']\n'
    print_file(str1)


def search_strength(data, flag, place, rule):
    findflag = False
    strength = []
    # look back
    for i in range(5):
        if data[place - i] not in rule:
            if flag[place - i] not in ['a', 'ad', 'nt', 'nz', 'j', 'b', 'm']:
                print_file(str(data[place - i]) + 'not in xxxx')
                break
            else:
                findflag = True
                print_file(str(data[place - i]) + 'in xxxx')
                strength.insert(0, data[place - i])

    # look forward
    if not findflag:
        for i in range(5):
            if data[place + i] not in rule:
                if flag[place + i] not in ['a', 'ad', 'nt', 'nz', 'j', 'b']:
                    print_file(str(data[place + i]) + 'not in xxxx')
                    break
                else:
                    print_file(str(data[place + i]) + 'in xxxx')
                    strength.append(data[place + i])
    return strength


if __name__ == '__main__':
    sentence = '国家励志奖学金 省级奖学金 专项奖学金 校内奖学金 SCIENCE NATURE SCIE EI ISTP SSCI A&HCI 重要核心刊物 跑步 高尔夫 羽毛球 羽毛球 游泳 乒乓球 冰球 篮球 旱冰 钢管舞 街舞 名族舞 芭蕾舞 桑巴舞 提琴 笛 琵琶 古筝 美声 通俗音乐 说唱 吉他 钢琴 贝斯 架子鼓 京剧 葫芦丝 乐器 素描 速写 油画 书法 水彩 水粉 陶艺 手工 摄影 插花 茶艺 编织 剪纸 小品 相声 主持 配音 话剧 文案 写作 学生会主席 副主席  班长 班干部 学习委员 生活委员 部长 秘书处 学习部 宣传部 文艺部 体育部 外联部 科技部 技术部 实践部 组织部 纪律部 社团部 建模竞赛  物理竞赛 数学竞赛 化学竞赛 综合能力竞赛 电子设计大赛 自动化大赛 制造挑战赛 科技竞赛 科技学术竞赛 汽车竞赛 机器人大赛  英语竞赛  创业大赛 演讲大赛 c语言 pascal basic vb ruby  java c++ dotnet .net c# html css html5 jquery php  jsp javascript python pearl spring ssh boot cloud android ios word 办公软件 excel 电子表格 ppt powerpoint  photoshop ps 图片处理 flash 视频处理 动画制作  英语四级 英语六级 日语 阿拉伯语 拉丁语 葡萄牙语 西班牙语 法语 德语 罗马尼亚语 雅思 IELTS 托福 TOEFL 剑桥商务英语 BEC 托业 TOEIC 专业英语八级 计算机一级 计算机二级 cfa 证券从业资格证 cfca afp cfp fsa fia ciia frm cfrm cpa acca cia cpv 善于交流沟通 性格外向  任劳任怨 听从领导安排 能吃苦 加班 敢打敢拼 上进心强 爱创新 新点子 发明创造  '

    print_file(str(preprocess(sentence)))
