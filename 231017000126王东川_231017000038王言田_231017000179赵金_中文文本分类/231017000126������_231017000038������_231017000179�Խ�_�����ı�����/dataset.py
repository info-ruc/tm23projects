import jieba, re, time, random, os
from langconv import Converter

class Dataset():
    def __init__(self):
        self.stopwords = self._get_all_stopwords()
    
    def _get_stopwords(self, path):
        with open(path, 'r', encoding='utf-8') as swf:
            stopwords = [i.strip() for i in swf.readlines()]
        return stopwords
    
    def _get_all_stopwords(self):
        stopwords_list = [
            'stopwords/cn_stopwords.txt',
            'stopwords/baidu_stopwords.txt',
            'stopwords/hit_stopwords.txt',
            'stopwords/scu_stopwords.txt'
        ]
        stopwords = []
        for path in stopwords_list:
            stopwords = stopwords + self._get_stopwords(path)
        stopwords = list(set(stopwords))
        print('load {} stopwords'.format(len(stopwords)))
        return stopwords
    
    def _preprocessing(self, sentence, cut_all=False):
        # 过滤掉特殊符号，只保留中文、英文、数字
        cleaner = re.compile(r"[^0-9a-zA-Z\u4e00-\u9fa5]+")
        sentence = cleaner.sub(' ', sentence)
        # 繁体字转化为简体字
        sentence = Converter('zh-hans').convert(sentence)
        # 使用jieba分词, 过滤停用词
        sentence = ' '.join([i for i in jieba.cut(sentence, cut_all=cut_all) if i.strip() and i not in self.stopwords])
        # print(sentence)
        return sentence

    def load_dataset(self, path):
        t1 = time.time()
        count = 0
        class_count = {}
        dataset = []
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            count = count + 1
            tmp = line.split('_!_')
            msg = {}
            msg['index'] = tmp[0]
            msg['class_id'] = tmp[1]
            msg['class_name'] = tmp[2]
            msg['rawdata'] = tmp[3]
            msg['data'] = self._preprocessing(tmp[3])
            if msg['class_id'] not in class_count.keys():
                class_count[msg['class_id']] = 1
            else:
                class_count[msg['class_id']] = class_count[msg['class_id']] + 1
            if count%10000 == 0:
                print('load {} news'.format(count))
            # print(msg)
            dataset.append(msg)
        t2 = time.time()
        print('load {} news, spend {:.3f}s'.format(len(dataset), t2-t1))
        for id in sorted(class_count.keys()):
            print('{}: {}'.format(id, class_count[id]))
        return dataset
    
    def split_train_test(self, dataset, train_ratio=0.8):
        train_set_size = int(len(dataset) * train_ratio)
        train_data = dataset[:train_set_size]
        test_data = dataset[train_set_size:]
        return train_data, test_data

    def write_train_test(self, train_data, test_data, dir='data'):
        train_path = os.path.join(dir, 'train.txt')
        test_path = os.path.join(dir, 'test.txt')
        with open(train_path, 'w', encoding='utf-8') as f:
            for i in train_data:
                f.write('{} __label__{}\n'.format(i['data'], i['class_id']))
        with open(test_path, 'w', encoding='utf-8') as f:
            for i in test_data:
                f.write('{} __label__{}\n'.format(i['rawdata'], i['class_id']))
        
    def run(self, input_path):
        dataset = self.load_dataset(input_path)
        random.seed(0)
        random.shuffle(dataset)
        train_data, test_data = self.split_train_test(dataset, train_ratio=0.8)
        self.write_train_test(train_data, test_data)


if __name__ == '__main__':
    data = Dataset()
    data.run('data/news_data.txt')
    
    # load 382688 news, spend 224.362s
    # 100: 6273
    # 101: 28031
    # 102: 39396
    # 103: 37568
    # 104: 27085
    # 106: 17672
    # 107: 35785
    # 108: 27058
    # 109: 41543
    # 110: 24984
    # 112: 21422
    # 113: 26909
    # 114: 340
    # 115: 19322
    # 116: 29300