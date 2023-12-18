# 简介
本项目主要使用Fasttext实现中文文本的快速分类。  
* Fasttext是一种使用子词的词向量训练方法，能够进行快速的文本分类。
* Fasttext综合深度学习的文本分类模型和机器学习的文本分类模型的优点，具有速度快、效果好、自动特征工程的优点。

# 运行环境
使用python3
依赖库安装
```
#安装jieba
pip install jieba
#安装fasttext，需要下载（https://www.lfd.uci.edu/~gohlke/pythonlibs/#fasttext）本地安装
pip install fasttext-0.9.2-cp37-cp37m-win_amd64.whl
```

# 数据准备
爬取头条新闻标题数据
```
python get_data.py
```
## 数据格式
```
6552431613437805063_!_102_!_news_entertainment_!_谢娜为李浩菲澄清网络谣言，之后她的两个行为给自己加分_!_佟丽娅,网络谣言,快乐大本营,李浩菲,谢娜,观众们
新闻ID      分类code        新闻字符串      新闻关键词
```
## 数据集统计
数据集共17个分类，382688条数据
| 种类id  | 种类  | 数量  |
|---------|-------|-------|
| 100 | 民生 故事 | 6273     |
| 101 | 文化 文化 | 28031     |
| 102 | 娱乐 娱乐 | 39396     |
| 103 | 体育 体育 | 37568     |
| 104 | 财经 财经 | 27085     |
| 105 | 时政 新时代 | 0     |
| 106 | 房产 房产 | 17672     |
| 107 | 汽车 汽车 | 35785     |
| 108 | 教育 教育 | 27058     |
| 109 | 科技 科技 | 41543     |
| 110 | 军事 军事 | 24984     |
| 111 | 宗教 无 | 0     |
| 112 | 旅游 旅游 | 21422     |
| 113 | 国际 国际 | 26909     |
| 114 | 证券 股票 | 340     |
| 115 | 农业 三农 | 19322     |
| 116 | 电竞 游戏 | 29300     |

# 数据预处理
进行数据清洗、分词、划分训练集和测试集等
```
python dataset.py
```
## 数据清洗
```
def _preprocessing(self, sentence, cut_all=False):
    # 过滤掉特殊符号，只保留中文、英文、数字
    cleaner = re.compile(r"[^0-9a-zA-Z\u4e00-\u9fa5]+")
    sentence = cleaner.sub(' ', sentence)
    # 繁体字转化为简体字
    sentence = Converter('zh-hans').convert(sentence)
    # 使用jieba分词, 过滤停用词
    sentence = ' '.join([i for i in jieba.cut(sentence, cut_all=cut_all) if i.strip() and i not in self.stopwords])
```
| 词表名 | 词表文件 |
| ------ | ------- |
| 中文停用词表                   | cn\_stopwords.txt    |
| 哈工大停用词表                 | hit\_stopwords.txt   |
| 百度停用词表                   | baidu\_stopwords.txt |
| 四川大学机器智能实验室停用词库 | scu\_stopwords.txt   |

## 数据预处理
```
def run(self, input_path):
    # 加载数据
    dataset = self.load_dataset(input_path)
    # 打乱顺序
    random.seed(0)
    random.shuffle(dataset)
    # 按照8:2划分训练集和测试集
    train_data, test_data = self.split_train_test(dataset, train_ratio=0.8)
    # 保存训练集数据和测试集数据
    self.write_train_test(train_data, test_data)
```
数据处理结果data/train.txt和data/test.txt
```
美国 打仗 钱 __label__110
彩票 一夜 暴富 方式 __label__109
火箭 112 102 爵士 圣保罗 41 分 火箭 晋级 西决 评价 保罗 本场 表现 __label__103
世界杯 百大 球星 当今 足坛 防守 后腰 __label__103
身在 农村 出路 __label__115
```

# 训练与测试
训练和测试
```
python train.py
```
## 训练模型
```
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

train('data/train.txt', model_path='model/last.model')
```
训练结果
```
    # train_precision: 0.9945092275028581
    # train_recall: 0.9945092275028581
    # Number of train examples: 306150
```
## 测试模型
```
def test(test_txt, model_path='model/last.model'):
    classifier = fasttext.load_model(model_path)
    test_result = classifier.test(test_txt)
    print('test_precision:', test_result[1])
    print('test_recall:', test_result[2])
    print('Number of test examples:', test_result[0])

test('data/test.txt', model_path='model/last.model')
```
测试结果
```
    # test_precision: 0.8788444955446968
    # test_recall: 0.8788444955446968
    # Number of test examples: 76538
```
## 预测
```
def predict(text, model_path='model/last.model'):
    classifier = fasttext.load_model(model_path)
    label = classifier.predict(text)
    print(label[0][0], label[1][0])
    
text = '美国 打仗 钱' # 110 军事 军事
predict(text, model_path='model/last.model')
```
测试结果
```
    # __label__110 0.9160338044166565
```

# 总结和展望
## 总结
Fasttext模型优点：  
* 速度非常快，并且效果还可以  
* 有开源实现，可以快速上手使用  
  
Fasttext模型缺点：  
* 模型结构简单，所以目前不是最优的模型  
* 因为使用词袋思想，所以语义信息获取有限  
  
## 展望
优化的方向
* 收集更丰富的数据集，均衡不同类型新闻的数量  
* 增加训练效果的图形化分析，如绘制准确率、召回率收敛的曲线等  
* 优化训练参数，如更换损失函数、wordNgrams、学习率lr等  
* 使用更复杂、更先进的模型  

# 合作与分工
## 小组成员
| 姓名 | 准考证号 | 专业 | 班级 |
| ---- | ------- | ---- | ---- |
| **王东川** | **231017000126**  | **计算机应用技术**  | **23春季班**  |
| **王言田** | **231017000038**  | **计算机应用技术**  | **22秋季班**  |
| **赵金** | **231017000179**  | **大数据科学与工程**  | **23春季班**  |
   
## 合作分工
| 姓名 | 分工 |
| ---- | ------- |
| **王言田** | 负责数据集准备、项目文档编写，参与代码编写和调试  |
| **赵金**   | 负责数据清洗、数据预处理，参与代码编写和调试、项目文档编写  |
| **王东川** | 负责运行环境准备、模型训练和测试、代码编写和调试，参与项目文档编写  |
