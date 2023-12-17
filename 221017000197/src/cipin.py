import jieba
import re

#打开要处理的文章
reader = open("221017000197\datasets\comment_1_fenci.txt",'r',encoding='utf8')
strs =reader.read()
result = open("221017000197\datasets\comment_1_fenci_cipin.txt","w",encoding='utf8')

# 分词，去重，列表
word_list = jieba.cut(strs,cut_all=True)
# 正则表达式去除数字，符号，单个字
new_words = []
for i in word_list:
    m = re.search("\d+",i)
    n = re.search("\W+",i)
    if not m and  not n and len(i)>1:
        new_words.append(i)

# 统计词频
word_count = {} # 创建字典
for i in set(new_words): # 用set去除list中的重复项
    word_count[i] = new_words.count(i)

# 格式整理
list_count = sorted(word_count.items(),key=lambda co:co[1],reverse=True)

# 打印结果
for i in range(300):
    print(list_count[i],file=result)

#关闭文件
reader.close()
result.close()
