import pickle
import pandas as pd
from random import shuffle
from operator import itemgetter
from collections import Counter, defaultdict

from config.params import TRAIN_FILE_PATH, TEST_FILE_PATH, NUM_WORDS


class FilePreprossing(object):
    def __init__(self, n):
        # 保留前n个高频字
        self.__n = n

    def _read_train_file(self):
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
        # sort_char_list = sorted(character_dict.items(), key=itemgetter(1), reverse=True)
        print(f'total {len(character_dict)} characters.')
        print('top 10 chars: ', sort_char_list[:10])
        # 保留前n个文字
        top_n_chars = [_[0] for _ in sort_char_list[:self.__n]]

        return label_list, top_n_chars

    def run(self):
        label_list, top_n_chars = self._read_train_file()
        with open('data/labels.pk', 'wb') as f:
            pickle.dump(label_list, f)

        with open('data/chars.pk', 'wb') as f:
            pickle.dump(top_n_chars, f)


def srcSplit(src_path, dst_path):
    fp = open(src_path, 'r')
    fp_split = open(dst_path, 'w')
    fp_split.write('label,content\n')

    lines = fp.readlines()
    for line in lines:
        label = line.split(',')[0]
        line = line.split(',')[1]
        count = 0
        total = len(line)
        while(total - count > 500):
            text = line[count:count+500]
            fp_split.write(label + ',' + text + '\n')
            count = count + 500


if __name__ == '__main__':
    srcSplit('data/train.csv', TRAIN_FILE_PATH)
    srcSplit('data/test.csv', TEST_FILE_PATH)

    processor = FilePreprossing(NUM_WORDS)
    processor.run()

    with open('data/labels.pk', "rb") as f:
        labels = pickle.load(f)
    print(labels)

    with open('data/chars.pk', "rb") as f:
        content = pickle.load(f)
    print(content)