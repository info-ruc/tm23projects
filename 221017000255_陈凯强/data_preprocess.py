import pandas as pd
from random import shuffle
from collections import Counter, defaultdict

from common import TRAIN_FILE_PATH, NUM_WORDS, LABELS_FILE_PATH, CHARS_FILE_PATH
from file_operation import savePkFile,readPkFile


def generatePkFile():
    train_pd = pd.read_csv(TRAIN_FILE_PATH)
    label_list = train_pd['label'].unique().tolist()
    # 统计文字频数
    character_dict = defaultdict(int)
    for content in train_pd['content']:
        for key, value in Counter(content).items():
            character_dict[key] += value
    # 不排序
    sort_char_list = [(k, v) for k, v in character_dict.items()]
    shuffle(sort_char_list)
    # 排序
    print(f'total {len(character_dict)} characters.')
    print('top 10 chars: ', sort_char_list[:10])
    # 保留前n个文字
    top_n_chars = [_[0] for _ in sort_char_list[:NUM_WORDS]]

    savePkFile(data=label_list, file_path=LABELS_FILE_PATH)
    savePkFile(data=top_n_chars, file_path=CHARS_FILE_PATH)


if __name__ == '__main__':
    generatePkFile()
    # 读取pickle文件
    labels = readPkFile(file_path=LABELS_FILE_PATH)
    print(labels)
    content = readPkFile(file_path=CHARS_FILE_PATH)
    print(content)
