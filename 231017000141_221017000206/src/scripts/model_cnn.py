import torch

from datainput import *
import torch.nn as nn
import torch.utils.data as Data


class TextCNN(nn.Module):
    def __init__(self, num_classes, vocab_size, cov_size=5, embedding_size=15, region_size=2, sequence_length=20, multi_conv=True):
        '''
        :param num_classes: 标签类别数
        :param vocab_size: 不同的词数
        :param cov_size: 卷积核数
        :param embedding_size: 词向量化大小
        :param region_size: 卷积核高度
        :param sequence_length: 序列（句子）长度
        :param multi_conv: 是否有多个不同大小的卷积核
        '''
        super(TextCNN, self).__init__()
        conv1_region_size = region_size
        conv2_region_size = region_size + 1
        conv3_region_size = region_size + 2
        self.multi_conv = multi_conv
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, cov_size, (conv1_region_size, embedding_size)),
            nn.ReLU(),
            nn.MaxPool2d((sequence_length + 1 - conv1_region_size, 1))
        )
        if multi_conv:
            self.conv2 = nn.Sequential(
                nn.Conv2d(1, cov_size, (conv2_region_size, embedding_size)),
                nn.ReLU(),
                nn.MaxPool2d((sequence_length + 1 - conv2_region_size, 1))
            )
            self.conv3 = nn.Sequential(
                nn.Conv2d(1, cov_size, (conv3_region_size, embedding_size)),
                nn.ReLU(),
                nn.MaxPool2d((sequence_length + 1 - conv3_region_size, 1))
            )
        if multi_conv:
            self.linear = nn.Linear(3 * cov_size, num_classes)
        else:
            self.linear = nn.Linear(cov_size, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x1 = self.conv1(x)
        if self.multi_conv:
            x2 = self.conv2(x)
            x3 = self.conv3(x)
            x = torch.cat([x1, x2, x3], dim=1)
        else:
            x = x1
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return x


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    test_inputs, test_labels, vocab_size = read_data(train=False)
    num_test = len(test_labels)
    test_labels, label2idx = label_digitize(test_labels)
    num_classes = len(label2idx)
    textcnn = TextCNN(num_classes, vocab_size, multi_conv=False)
    test_inputs = torch.LongTensor(test_inputs)
    test_labels = torch.LongTensor(test_labels)
    dataset = Data.TensorDataset(test_inputs, test_labels)
    data_loader = Data.DataLoader(dataset, 4, True)
    right_num = 0
    for i, (x, y) in enumerate(data_loader):
        y_pre = textcnn(x)
        # print(y_pre)
        # print(y)
        # print(y_pre.shape)
        # print(y.shape)
        right_num += sum(torch.argmax(y_pre, dim=1) == y)

    print(right_num, num_test, right_num/num_test)

