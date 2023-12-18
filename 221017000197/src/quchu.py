import jieba

# jieba.load_userdict('userdict.txt')
# 创建停用词list
targetTxt = '221017000197\datasets\ctingyong_2.txt'
def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords

def seg_sentence(sentence):
    sentence_seged = jieba.cut(sentence.strip())
    stopwords = stopwordslist(targetTxt)  # 这里加载停用词的路径
    outstr = ''
    for word in sentence_seged:
        if word not in stopwords:
            if word != '\t':
                outstr += word
                outstr += " "
    return outstr

inputs = open('221017000197\datasets\comment_1.txt', 'r', encoding='utf-8')

outputs = open('221017000197\datasets\comment_1_fenci.txt', 'w',encoding='utf-8')
for line in inputs:
    line_seg = seg_sentence(line)  # 这里的返回值是字符串
    outputs.write(line_seg + '\n')
outputs.close()
inputs.close()
