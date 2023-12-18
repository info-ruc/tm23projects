import os
import jieba
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import RandomizedSearchCV,train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from scipy.stats import uniform

class EmailToWordFeatures:

index_file= '.\\datasets\\trec06c\\full\\index'
stop_word_file = '.\\datasets\\trec06c\\chinese_stop_vocab.txt'
ham_file = '.\\datasets\\trec06c\\ham_data'
spam_file = '.\\datasets\\trec06c\\spam_data'

email_to_features = EmailToWordFeatures(stop_word_file=stop_word_file)
ham_words = email_to_features.get_email_words(ham_file)
spam_words = email_to_features.get_email_words(spam_file)
print('ham email numbers:',len(ham_words))
print('spam email numbers:',len(spam_words))

ham email numbers: 600
spam email numbers: 600

all_email = []
all_email.extend(ham_words)
all_email.extend(spam_words)
print('all test email numbers:',len(all_email))
words_to_matrix = email_to_features.text_to_feature_matrix(all_email)
print(words_to_matrix)
all test email numbers: 1200
(1200, 22556)

      故事  领导  回到  儿子  感情  有个  大概  民俗  出国  教育  ...  培训网  商友会  网管  埃森哲  驱鼠器  条例  \
0      1   2   1   1   1   1   1   1   1   1  ...    0    0   0    0    0   0   
1      0   0   0   0   5   0   0   0   0   0  ...    0    0   0    0    0   0   
2      0   0   0   0   0   0   0   0   0   0  ...    0    0   0    0    0   0   
3      0   0   0   0   0   0   0   0   0   0  ...    0    0   0    0    0   0   
4      0   0   0   0   0   0   0   0   0   0  ...    0    0   0    0    0   0   
...   ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ...  ...  ...  ..  ...  ...  ..   
1195   0   0   0   0   0   0   0   0   0   0  ...    0    0   0    0    0   0   
1196   0   0   0   0   0   0   0   0   0   0  ...    0    0   0    0    0   0   
1197   0   0   0   0   0   0   0   0   0   0  ...    0    0   0    0    0   0   
1198   0   0   0   0   0   0   0   0   0   0  ...    0    0   0    0    0   0   
1199   0   0   0   0   0   0   0   0   0   0  ...    0    0   0    0    0   0   

      智囊  教练  含双早  王府井  
0      0   0    0    0  
1      0   0    0    0  
2      0   0    0    0  
3      0   0    0    0  
4      0   0    0    0  
...   ..  ..  ...  ...  
1195   0   0    0    0  
1196   0   0    0    0  
1197   0   0    0    0  
1198   0   0    0    0  
1199   0   0    0    0  

[1200 rows x 3099 columns]

label_matrix = np.zeros((len(all_email),1))
label_matrix[0:len(ham_words),:] = 1 
x_train,x_test,y_train,y_test = train_test_split(words_to_matrix,label_matrix,test_size=0.2,random_state=42)

svc = LinearSVC(loss='hinge',dual=True)
param_distributions = {'C':uniform(0,10)}
rscv_clf =RandomizedSearchCV(estimator=svc, param_distributions=param_distributions,cv=3,n_iter=200,verbose=2)
rscv_clf.fit(x_train,y_train)
print('best_params:',rscv_clf.best_params_)
Fitting 3 folds for each of 200 candidates, totalling 600 fits
[CV] C=6.119041659192192 .............................................
[CV] .............................. C=6.119041659192192, total=   0.0s
[CV] C=6.119041659192192 .............................................
[CV] .............................. C=6.119041659192192, total=   0.1s
[CV] C=6.119041659192192 .............................................
[CV] .............................. C=6.119041659192192, total=   0.1s
[CV] C=6.103402593686549 .............................................
...
...
...
[CV] .............................. C=4.395657632563425, total=   0.2s
best_params: {'C': 0.0279898379592336}

y_prab = rscv_clf.predict(x_test)
print('accuracy:',accuracy_score(y_prab,y_test))
accuracy: 0.9791666666666666

#分别选择一封正式邮件和垃圾邮件进行
正式邮件内容如下：

很久以前，我为了考人大，申请了他的ID，而现在却不对外开放了。
申请水木的ID，真的是不知道出于什么缘故。离开校园寻找一份校园的感觉，怀着对清华的向往，还是为了偶尔无聊工作的一些调剂……
我讨厌多愁善感，却时常沉浸于其中，生活中的挫折，不幸，让我知道自己得坚强……
可每天的灰色心情却挥之不去，我可以把自己的心事埋于深处，笑着对待我身边的每一个人，告诉我乐观。身边最亲的人，也许知道或不知道我的脆弱和恐惧。而唯一知道的人，告诉我“希望你坚不可摧”。
回想多年前，为“在靠近的地方住下，能掩耳不听烦世喧嚣，要一份干净的自由自在”而感动，那，是否是对今天的预见，无知是快乐的，而不知道责任也是快乐的。我可以逃避一时，却始终要面对。
垃圾邮件如下：

这是一封善意的邮件,如给你造成不便,请随手删除.SOHO建站代理网诚聘兼职网站代理　
一、职业要求:
１、有上网条件(在家中、办公室、网吧等地)；
２、每天能有１－２小时上网时间；
３、有网络应用的基础（会上论坛发贴子、发电子邮件,
与客户QQ沟通等）。
二、工作任务:
您报名加入我公司后，公司将分配给您一个属
于自己的冠名为SOHO致富联盟的网站，作为站长，您的任
务就是利用各种方法宣传这个网站，让客户从你的网站上
购买更多的商品，并接受你的建议，也同意加盟SOHO建站
代理网网站的兼职代理，这样你便拥有滚滚不断的财源。
三、工资待遇:3000元以上／月。业绩累积,收入直线上升.

def email_to_predict_matrix(words,features):
    cv = CountVectorizer()
    words_to_vect = cv.fit_transform(words)
    words_to_marix = pd.DataFrame(words_to_vect.toarray())
    vocabulary = cv.vocabulary_
    
    words_numbers_list = [] 
    for feature in features:
        if feature in cv.vocabulary_.keys():
            words_numbers_list.append(words_to_marix[vocabulary[feature]][0])
        else:
            words_numbers_list.append(0)
    words_numbers_matrix = pd.DataFrame([words_numbers_list],columns = features)
    return words_numbers_matrix

valid_ham_email = '.\\datasets\\trec06c\\valid_ham_email'
valid_spam_email = '.\\datasets\\trec06c\\valid_spam_email'

email_to_features_valid = EmailToWordFeatures(stop_word_file=stop_word_file)
valid_ham_email_words = email_to_features_valid.get_email_words(valid_ham_email)
valid_spam_email_words = email_to_features_valid.get_email_words(valid_spam_email)

valid_ham_words_maxtrix = email_to_predict_matrix(valid_ham_email_words,words_to_matrix.columns)
valid_spam_words_maxtrix = email_to_predict_matrix(valid_spam_email_words,words_to_matrix.columns)

print('测试正式邮件----------')
print('预测结果：',rscv_clf.predict(valid_ham_words_maxtrix))
测试正式邮件----------
预测结果： [1.]

print('测试垃圾邮件----------')
print('预测结果：',rscv_clf.predict(valid_spam_words_maxtrix))

测试垃圾邮件----------
预测结果： [0.]

#改进计划
将特征词矩阵改word2vect

