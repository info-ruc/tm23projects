# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import sys
import os
import argparse
from utils.dictionary import *
from utils.model import *
from utils.process import *

sys.path.append(os.getcwd())
parser = argparse.ArgumentParser()

parser.add_argument('--train_file', '-tr', type=str, default='./data/atis.train.txt',
                    help='训练数据文件所处的路径.')
parser.add_argument('--test_file', '-te', type=str, default='./data/atis.test.txt',
                    help='测试数据文件所处的路径.')
parser.add_argument('--all_file', '-al', type=str, default='./data/atis.all.txt',
                    help='全体数据文件所处的路径.')
parser.add_argument('--dev_file', '-de', type=str, default='./data/atis.dev.txt',
                    help='最终评测模型文件所处的路径.')
parser.add_argument('--optimizer', '-op', type=str, default='adagrad',
                    help='选择优化算法, 只能选择 sgd, adam 和 adagrad.')
parser.add_argument('--batch_size', '-bs', type=int, default=32,
                    help='每轮训练 batch 的大小, 推荐 32.')
parser.add_argument('--learning_rate', '-lr', type=float, default=1e-1,
                    help='训练的初始学习率, 推荐 0.1.')
parser.add_argument('--num_epoch', '-ne', type=int, default=800,
                    help='训练过程的循环次数, 推荐 800.')
parser.add_argument('--print_each', '-pe', type=int, default=10,
                    help='训练过程打印 batch 信息的间隔, 推荐10.')
parser.add_argument('--save_model', '-sm', type=int, default=50,
                    help='训练过程中保存 encoder 和 decoder 模型的间隔, 推荐 50.')
parser.add_argument('--validate_each', '-ve', type=int, default=30,
                    help='训练过程中检验测试集得分的间隔, 最好不要小于 10.')
parser.add_argument('--model_save', '-ms', type=str, default='./save/model/',
                    help='训练过程中保存模型的地址.')
parser.add_argument('--word_embedding', '-we', type=int, default=200,
                    help='嵌入词向量维度的大小, 这里使用 200 以上容易超内存.')
parser.add_argument('--slot_embedding', '-se', type=int, default=12,
                    help='序列标注 slot 的嵌入词向量大小, 最好不要比 12 更低.')
parser.add_argument('--num_layers', '-nl', type=int, default=1,
                    help='在 Encoder 中 LSTM 的层数, 论文中仅 1 层.')
parser.add_argument('--bidirectional', '-bi', type=bool, default=True,
                    help='在 Encoder 中 LSTM 是否双向, 在论文中使用双向.')
parser.add_argument('--random_state', '-rs', type=int, default=20,
                    help='随机化训练数据序列的伪随机种子.')
parser.add_argument('--hidden_size', '-hz', type=int, default=128,
                    help='在 Encoder 和 Decoder 中 LSTM 隐层维度的大小.')
parser.add_argument('--is_test', '-t', type=bool, default=True,
                    help='测试模式.')
args = parser.parse_args()

train_file = args.train_file
test_file = args.test_file
all_file = args.all_file
dev_file = args.dev_file
optimizer = args.optimizer
batch_size = args.batch_size
learning_rate = args.learning_rate
num_epoch = args.num_epoch
print_each = args.print_each
save_model = args.save_model
validate_each = args.validate_each
model_save = args.model_save
word_embedding = args.word_embedding
slot_embedding = args.slot_embedding
num_layers = args.num_layers
bidirectional = args.bidirectional
random_state = args.random_state
hidden_size = args.hidden_size

if __name__ == "__main__":

    print('命令参数格式解析正确. 使用加 Attention 机制的 Encoder-Decoder 模型训练 SLU.')
    print('其中 Encoder 是一个 {} 层的 双向(是否:{}) 的LSTM, 而 Decoder 是一个单向单层的 LSTM.'.format(num_layers,
                                                                                                       bidirectional))
    print()
    print('使用初始学习率为 {:.6f} 的 {} 优化算法训练模型, 共训练 {} 轮, 每轮 batch 大小为 {}.'.format(learning_rate,
                                                                                                       optimizer,
                                                                                                       num_epoch,
                                                                                                       batch_size))
    print('其中, 每 {} 轮训练打印 batch 的训练信息, 每 {} 轮训练打印测试集的精确度, 包括 Accuracy 和 F1,'.format(
        print_each, validate_each))
    print(
        '每 {} 轮训练保存模型. 此外, 使用嵌入词向量维度为 {}, 标注嵌入向量维度为 {}.'.format(save_model, word_embedding,
                                                                                             slot_embedding))
    print()

    data = Dataset(random_state=random_state)

    time_start = time.time()
    data.quick_build(train_path=train_file,
                     test_path=test_file,
                     all_path=all_file,
                     dev_path=dev_file)
    print('训练集, 测试集, 全集数据, 开发集全部读取完毕, 共耗时 {:.6} 秒.\n'.format(time.time() - time_start))
    # data.quick_build()
    if args.is_test:
        encoder = torch.load('./save/model/encoder.pth')
        decoder = torch.load('./save/model/decoder.pth')
        devset_evaluation(encoder, decoder, data)
        print("test over !")
        exit(1)
    word_dict, label_dict, intent_dict = data.get_alphabets()

    encoder = Encoder(len(word_dict), word_embedding, hidden_size, num_layers, bidirectional)
    decoder = Decoder(hidden_size, slot_embedding, len(label_dict), len(intent_dict))

    print(len(data.get_dataset('dev')['sentence_list']))

    train(encoder, decoder, data, optimizer,
          batch_size, learning_rate,
          num_epoch, print_each, save_model, validate_each,
          model_save)

    predict(data, encoder=encoder, decoder=decoder, name='self-test', give_predictions=2)
    predict(data, encoder=encoder, decoder=decoder,
            sample_tuple=data.get_dev(), name='self-dev', give_predictions=1)

    #
