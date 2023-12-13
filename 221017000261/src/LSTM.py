import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import nltk
from nltk.corpus import stopwords
import numpy as np

# 下载停用词s
nltk.download('stopwords')
# 获取英文停用词列表
stop_words = set(stopwords.words('english'))
# 打印英文停用词列表
print('英文停用词数量:', len(stop_words))
print('英文停用词：\n', stop_words)

# 加载 IMDb 数据集，如果本地没有该数据集，会自动下载
(train_data, test_data), info = tfds.load('imdb_reviews', split=('train', 'test'), as_supervised=True, with_info=True)

# 创建 Tokenizer 对象
tokenizer = Tokenizer()

# 创建文本序列和标签列表
train_texts = []
train_labels = []
test_texts = []
test_labels = []

# 获取训练集和测试集的文本以及对应的标签
for text, label in train_data:
    train_texts.append(str(text.numpy()))
    train_labels.append(label.numpy())

for text, label in test_data:
    test_texts.append(str(text.numpy()))
    test_labels.append(label.numpy())

# 将文本转换为标记化的序列
tokenizer.fit_on_texts(train_texts)
word_index = tokenizer.word_index
word_index['<UNK>'] = len(word_index)+1

# 去除停用词并将文本转换为序列
def preprocess_text(texts):
    sequences = tokenizer.texts_to_sequences(texts)
    filtered_sequences = []
    for sequence in sequences:
        filtered_sequence = [word_index['<UNK>'] for word in sequence if word not in stop_words]#[word_index[word] for word in sequence if word not in stop_words]
        filtered_sequences.append(filtered_sequence)
    return filtered_sequences

# 对训练集和测试集进行预处理
train_sequences = preprocess_text(train_texts)
test_sequences = preprocess_text(test_texts)

# 设置序列最大长度
max_length = 200
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding='post', truncating='post')
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding='post', truncating='post')

# 转换为 NumPy 数组
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

# 构建模型
model = Sequential([
    Embedding(len(word_index) + 1, 64, input_length=max_length),
    LSTM(64),
    Dense(1, activation='sigmoid')
])

# 编译模型
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型
num_epochs = 3
model.fit(train_padded, train_labels, epochs=num_epochs, validation_data=(test_padded, test_labels))

# 进行预测并输出置信度
predictions = model.predict(test_padded)
confidence = predictions.squeeze()  # 模型预测的置信度
print("测试集前10个样本的预测结果和置信度：")
for i in range(10):
    print(f"样本 {i + 1}: 预测结果: {predictions[i]}, 置信度: {confidence[i]}")