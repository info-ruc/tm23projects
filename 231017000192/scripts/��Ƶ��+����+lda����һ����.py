# -*- coding: utf-8 -*-

import pandas as pd
import jieba
import re
from collections import Counter
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']    # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False    # 用来正常显示负号
import warnings
warnings.filterwarnings('ignore')
from wordcloud import WordCloud
from PIL import Image
import numpy as np
from gensim import corpora, models
from gensim.models import CoherenceModel
# 读取Excel文件
df = pd.read_excel('流浪地球2影评.xlsx')

# 处理缺失值和重复值
df.dropna(inplace=True)
df.drop_duplicates(subset=['影评'], inplace=True)
# 数据清洗
df['文本内容'] = df['影评'].apply(lambda x: re.sub(r'[^\u4e00-\u9fa5]+', '', str(x)))

# 将文本分词并去除停用词和长度为1的字符
# 创建停用词列表
def stopwordslist():
    stopwords = [line.strip() for line in open('stopwords_list.txt',encoding='UTF-8').readlines()]
    return stopwords

stopwords = stopwordslist()
df['分词列表'] = df['文本内容'].apply(lambda x: [word for word in jieba.cut(x) if word not in stopwords and len(word) > 1])

# 统计高频词并画条形图
words = []
for word_list in df['分词列表']:
    words.extend(word_list)
word_counts = Counter(words)
top_word_counts = word_counts.most_common(50)

x_labels = []
y_values = []
for word, count in top_word_counts:
    x_labels.append(word)
    y_values.append(count)
plt.figure(figsize=(20,10))
plt.bar(x_labels, y_values)
plt.xlabel('Top Words')
plt.ylabel('Word Counts')
plt.xticks(rotation=90)
plt.show()

# 绘制高频词词云图
mask = np.array(Image.open('词云图底板.jpg'))
wordcloud = WordCloud(scale=4,
    background_color='white', 
    mask=mask, 
    colormap='tab10',
    font_path='C:/Windows/Fonts/SIMYOU.TTF').generate_from_frequencies(word_counts)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# 计算主题寻优曲线-一致性得分
dictionary = corpora.Dictionary(df['分词列表'])
corpus = [dictionary.doc2bow(text) for text in df['分词列表']]
'''
主题寻优曲线可以帮助我们确定最佳的主题数。在主题寻优曲线上，横轴表示主题数，纵轴表示主题一致性得分。一致性得分越高，表示主题之间的关联性越强，主题分析结果越好。

通常，我们可以选择一致性得分最高的主题数作为最佳主题数。但是，在实际应用中，最佳主题数还需考虑业务需求和实际效果。如果主题数过少，可能会导致主题之间的关联性不够强，无法发现主题之间的细节和差异。如果主题数过多，可能会导致主题之间的关联性过于复杂，难以解释和理解。

因此，我们需要综合考虑业务需求和实际效果，选择最佳的主题数。可以通过多次运行LDA模型，选取不同的主题数，比较主题分析结果，最终确定最佳的主题数。
'''
coh_values = []
for num_topics in range(2, 11):
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary)
    coherence_model_lda = CoherenceModel(model=lda_model, texts=df['分词列表'], dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    coh_values.append(coherence_lda)

# 绘制主题寻优曲线
plt.plot(range(2, 11), coh_values)
plt.xlabel('Number of Topics')
plt.ylabel('Coherence Score')
plt.show()

# 计算主题寻优曲线-困惑度曲线
# perplexity_values = []
# for num_topics in range(2, 11):
#     lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary)
#     perplexity_model_lda = lda_model.log_perplexity(corpus)
#     perplexity_lda = np.exp2(-perplexity_model_lda)
#     perplexity_values.append(perplexity_lda)


# plt.plot(range(2, 11), perplexity_values)
# plt.xlabel('Number of Topics')
# plt.ylabel('Perplexity Score')
# plt.show()





# 构建LDA模型

num_topics = coh_values.index(max(coh_values)) + 2
lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary)

# 输出LDA主题分布并将主题词导出到Excel
topic_words = []
for topic in lda_model.print_topics(num_topics=num_topics, num_words=20):
    topic_words.append(re.findall(r'"([^"]*)"', topic[1]))
    print(topic)

df_topics = pd.DataFrame(topic_words,columns=['主题词' + str(i+1) for i in range(20)],
                         index=['主题' + str(i+1) for i in range(num_topics)])
df_topics.to_excel('LDA主题提取结果.xlsx')



# 主题气泡图        
import pyLDAvis.gensim        
vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)
pyLDAvis.show(vis)
