import os


# 项目文件设置
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
TRAIN_FILE_PATH = os.path.join(PROJECT_PATH, 'dataset/train.csv')
TEST_FILE_PATH = os.path.join(PROJECT_PATH, 'dataset/test.csv')
LABELS_FILE_PATH = './dataset/labels.pk'
CHARS_FILE_PATH = './dataset/chars.pk'
MODEL_FILE_PATH = './model/news_text_cl_cn1.pth'

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
EPOCHS = 10
