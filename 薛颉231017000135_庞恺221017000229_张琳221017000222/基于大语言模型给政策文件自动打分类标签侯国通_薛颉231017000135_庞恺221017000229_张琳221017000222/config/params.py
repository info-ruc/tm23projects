import os

# 项目文件设置
TRAIN_FILE_PATH = 'data/train_split.csv'
TEST_FILE_PATH = 'data/test_split.csv'
MODEL_PATH = 'weights/best.pth'

# 预处理设置
NUM_WORDS = 5500
PAD = '<PAD>'
PAD_NO = 0
UNK = '<UNK>'
UNK_NO = 1
START_NO = UNK_NO + 1
SENT_LENGTH = 200

# 模型参数
EMBEDDING_SIZE = 300
TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 40
