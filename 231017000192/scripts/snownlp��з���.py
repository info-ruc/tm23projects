# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import re
import jieba
from snownlp import SnowNLP
import matplotlib.pyplot as plt

# 读取excel指定列
df = pd.read_excel('流浪地球2影评.xlsx', usecols=['影评'])

# 缺失值处理
df.dropna(inplace=True)

# 去重
df.drop_duplicates(inplace=True)

# 使用Snownlp打分
def get_score(text):
    s = SnowNLP(str(text))
    return s.sentiments

df['得分'] = df['影评'].apply(get_score)

# 标签处理
def get_label(score):
    if score >= 0.5:
        return 'positive'
    else:
        return 'negative'

df['sentiment_key'] = df['得分'].apply(get_label)

# 情绪类别分布饼图
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize = (10,6))
s_label= np.array(df["sentiment_key"].value_counts())
plt.pie(s_label,
        labels=['positive','negative'], # 设置饼图标签
        colors=["gold", "lightskyblue"], # 设置饼图颜色
        autopct='%.2f%%', # 格式化输出百分比
        explode = [0, 0.1] #每块分裂的距离
       )
plt.title("情绪类别分布")
plt.show()



# 正负情感均值
posneg = df.groupby(["sentiment_key"])['得分'].mean()

# 计算要显示的数字
pos_label = format(posneg.values[0], '.2f') # 保留两位小数
neg_label = format(posneg.values[1], '.2f') # 保留两位小数

# 添加柱状图和文本注释
plt.bar(posneg.index[0],posneg.values[0],hatch='--',color='gold')
plt.text(posneg.index[0], posneg.values[0], pos_label, ha='center', va='bottom')

plt.bar(posneg.index[1],posneg.values[1],hatch='\\',color='dodgerblue')
plt.text(posneg.index[1], posneg.values[1], neg_label, ha='center', va='bottom')
plt.title('情感得分')
plt.xlabel('情感类别')
plt.ylabel('平均情感分值')

# 情感分直方图
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

bins=np.arange(0,1.1,0.1)
plt.hist(df['得分'],bins,color='#4F94CD',alpha=0.9)
plt.xlim(0,1)
plt.xlabel('情感分')
plt.ylabel('数量')
plt.title('情感分直方图')
plt.show()
