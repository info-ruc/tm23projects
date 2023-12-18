import fasttext

def train(train_txt, model_path='model/last.model'):
    classifier = fasttext.train_supervised(
        train_txt,         # 训练数据
        label='__label__',  #标签前缀
        dim=100,            # 词向量大小
        epoch=100,          # 训练次数
        lr=0.5,             # 学习率
        loss='softmax',     # 损失函数 ns, hs, softmax, ova
        wordNgrams=2,       # 最大词长度ngram
        ws=5               # 上下文窗口的大小
    )
    classifier.save_model(model_path)
    train_result = classifier.test(train_txt)
    print('train_precision:', train_result[1])
    print('train_recall:', train_result[2])
    print('Number of train examples:', train_result[0])
    
def test(test_txt, model_path='model/last.model'):
    classifier = fasttext.load_model(model_path)
    test_result = classifier.test(test_txt)
    print('test_precision:', test_result[1])
    print('test_recall:', test_result[2])
    print('Number of test examples:', test_result[0])

def predict(text, model_path='model/last.model'):
    classifier = fasttext.load_model(model_path)
    label = classifier.predict(text)
    print(label[0][0], label[1][0])


if __name__ == '__main__':
    train('data/train.txt', model_path='model/last.model')
    test('data/test.txt', model_path='model/last.model')
    
    text = '美国 打仗 钱' # 110 军事 军事
    predict(text, model_path='model/last.model')
    
    # train_precision: 0.9945092275028581
    # train_recall: 0.9945092275028581
    # Number of train examples: 306150
    
    # test_precision: 0.8788444955446968
    # test_recall: 0.8788444955446968
    # Number of test examples: 76538
    
    # __label__110 0.9160338044166565
    