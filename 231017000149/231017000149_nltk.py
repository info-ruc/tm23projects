#!/usr/local/python3.6
#import nltk
#nltk.download('movie_reviews')
#nltk.download('punkt')
#nltk.download('omw-1.4')
#nltk.download('stopwords')
#nltk.download('wordnet')
import os
import nltk
path = "/root"
os.chdir(path)
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.probability import FreqDist
# 分词
def tokenize(text):
    tokens = word_tokenize(text)
    return tokens

# 去除停用词
def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    return filtered_tokens

# 词形还原
def lemmatize(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens

# 对整个数据集进行预处理
"""
documents = []
for fileid in movie_reviews.fileids():
    words = tokenize(movie_reviews.raw(fileid))
    words = remove_stopwords(words)
    words = lemmatize(words)
    documents.append((words, movie_reviews.categories(fileid)))
"""
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]    
#使用朴素贝叶斯分类器进行训练和评估：
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle


# 将数据集划分为训练集和测试集
X, y = zip(*documents)
num_train_sets = 20
for i in range(num_train_sets):
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,random_state=42, shuffle=False, stratify=y)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,stratify=y)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#print(X_train,X_test)
	X_train = [str(x) for x in X_train]
	X_test = [str(x) for x in X_test]

# 将文本转换为特征向量
	vectorizer = TfidfVectorizer()
#vectorizer = CountVectorizer()
	X_train_vec = vectorizer.fit_transform(X_train)
	X_test_vec = vectorizer.transform(X_test)

# 使用朴素贝叶斯分类器进行训练
	clf = MultinomialNB()
	clf.fit(X_train_vec, y_train)

# 预测测试集
	y_pred = clf.predict(X_test_vec)
#	print(y_pred)

# 计算准确率
	accuracy = accuracy_score(y_test, y_pred)
	print("Accuracy: ", accuracy)


#新评论数据测试
X_test = ["this is a very bad films", "This movie was truly a cinematic masterpiece. From the plot to the character development, and even the visual effects, every aspect was filled with innovation and surprises. I particularly admired how the director skillfully incorporated complex emotional threads into the story, making the audience both nervous and excited while watching. The performances of the actors were also outstanding, as they successfully portrayed unique and relatable characters. The musical score further enhanced the film, adding more drama to the plot's progression. Overall, it is a well-crafted and captivating movie that is definitely worth watching again and again."]

# 将测试数据转换为特征向量
X_test_vec = vectorizer.transform(X_test)

# 使用模型进行预测
print('对输入的新评论进行预测')
predictions = clf.predict(X_test_vec)
print(predictions)
