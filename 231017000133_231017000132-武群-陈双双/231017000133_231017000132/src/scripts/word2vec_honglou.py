# 开发团队：铁科院经纬公司
# 开发人员：wuqun
# 开发时间：2023-12-2 21:12

import jieba
import re
import warnings
from gensim.models import Word2Vec

import numpy as np
from sklearn.decomposition import PCA


import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

with open("honglou.txt", 'r', encoding='utf-8') as f:  # 读入文本
    lines = []
    for line in f:  # 分别对每段分词
        temp = jieba.lcut(line)  # 结巴分词 精确模式
        words = []
        for i in temp:
            # 过滤掉所有的标点符号
            i = re.sub("[\s+\.\!\/_,$%^*(+\"\'””《》]+|[+——！，。？、~@#￥%……&*（）：；‘]+", "", i)
            if len(i) > 0:
                words.append(i)
        if len(words) > 0:
            lines.append(words)
print(lines[0:5])  # 预览前5行分词结果

# 调用Word2Vec训练 参数：size: 词向量维度；window: 上下文的宽度，min_count为考虑计算的单词的最低词频阈值
model = Word2Vec(lines, vector_size=20, window=2, min_count=3, epochs=7, negative=10, sg=1)
print("的词向量：\n", model.wv.get_vector('晴雯'))
print("\n和晴雯相关性最高的前20个词语：")
# model.wv.most_similar('孔明', topn = 20)# 与孔明最相关的前20个词语
print(model.wv.most_similar('晴雯', topn=20))  # 与孔明最相关的前20个词语

# 将词向量投影到二维空间
rawWordVec = []
word2ind = {}
for i, w in enumerate(model.wv.index_to_key):
    rawWordVec.append(model.wv[w])  # 词向量
    word2ind[w] = i  # {词语:序号}
rawWordVec = np.array(rawWordVec)
X_reduced = PCA(n_components=2).fit_transform(rawWordVec)  # PCA降2维


plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示
plt.rcParams['axes.unicode_minus'] = False  # 解决符号无法显示
# 绘制星空图
# 绘制所有单词向量的二维空间投影
fig = plt.figure(figsize=(15, 10))
ax = fig.gca()
ax.set_facecolor('white')
ax.plot(X_reduced[:, 0], X_reduced[:, 1], '.', markersize=1, alpha=0.3, color='black')

# 绘制几个特殊单词的向量
words = ['贾宝玉', '王熙凤', '迎春', '探春', '林黛玉', '晴雯', '史湘云']

for w in words:
    if w in word2ind:
        ind = word2ind[w]
        xy = X_reduced[ind]
        plt.plot(xy[0], xy[1], '.', alpha=1, color='orange', markersize=10)
        plt.text(xy[0], xy[1], w, alpha=1, color='red')
