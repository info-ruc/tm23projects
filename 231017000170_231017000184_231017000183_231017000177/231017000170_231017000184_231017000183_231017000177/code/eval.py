from importlib import import_module
import torch
from sklearn import metrics
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

vocab_size = 5000
batch_size = 128
max_length = 32
embed_dim = 300
label_num = 10
epoch = 5
eval_path = '../data/dev.txt'
vocab_path = '../data/vocab.txt'

output_path = '../output/'
model_path = '../output/model.pt'


def get_data(path):
    input_vocab = open(vocab_path, 'r', encoding='utf-8')
    vocabs = {}
    for item in input_vocab.readlines():
        word, wordid = item.replace('\n', '').split('\t')
        vocabs[word] = int(wordid)
    input_data = open(path, 'r', encoding='utf-8')
    x = []
    y = []
    for item in input_data.readlines():
        sen, label = item.replace('\n', '').split('\t')
        tmp = []
        for item_char in sen:
            if item_char in vocabs:
                tmp.append(vocabs[item_char])
            else:
                tmp.append(1)
            if len(tmp) >= max_length:
                break
        x.append(tmp)
        y.append(int(label))

    # padding
    for item in x:
        if len(item) < max_length:
            item += [0] * (max_length - len(item))

    label_num = len(set(y))
    # x_train, x_test, y_train, y_test = train_test_split(np.array(x), np.array(y), test_size=0.2)
    x = np.array(x)
    print(x.shape)
    y = np.array(y)
    return x, y, label_num

class DealDataset(Dataset):
    def __init__(self, x_train, y_train, device):
        self.x_data = torch.from_numpy(x_train).long().to(device)
        self.y_data = torch.from_numpy(y_train).long().to(device)
        self.len = x_train.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

def evaluate(model, dataloader_dev):
    model.eval()
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for datas, labels in dataloader_dev:
            output = model(datas)
            predic = torch.max(output.data, 1)[1].cpu()
            predict_all = np.append(predict_all, predic)
            labels_all = np.append(labels_all, labels.cpu())
            if len(predict_all) > 1000:
                break
    acc = metrics.accuracy_score(labels_all, predict_all)
    return acc


if __name__ == "__main__":
    debug = False
    # model_name = 'TextCNN'
    model_name = 'Transformer'
    # model_name = ‘TextLSTM’
    module = import_module(model_name)
    config = module.Config(vocab_size, embed_dim, label_num)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = module.Model(config).to(device)
    if debug:
        # 维度：batch_size * max_length, 数值：0~200之间的整数，每一行表示wordid
        inputs = torch.randint(0, 200, (batch_size, max_length))
        # 维度：batch_size * 1， 数值：0~2之间的整数，维度扩充1，和input对应
        labels = torch.randint(0, 2, (batch_size, 1)).squeeze(0)
        print(model(inputs))
    else:
        x_eval, y_eval, label_num = get_data(eval_path)
        # x_train, y_train, label_num = get_data(train_path)
        dataset = DealDataset(x_eval, y_eval, device)
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
        # torch.save(model, 'model.pt')
        model.load_state_dict(torch.load(model_path))
        # model.load(torch.load(model_path))
        model.eval()
        predict_all = np.array([], dtype=int)
        labels_all = np.array([], dtype=int)
        score_all = np.array([], dtype=float)
        with torch.no_grad():
            for datas, labels in tqdm(dataloader):
                output = model(datas)
                # print(torch.softmax(output.data, 1))
                score_all = np.append(score_all, torch.softmax(output.data, 1).cpu().numpy())
                predic = torch.max(output.data, 1)[1].cpu().numpy()
                loss = F.cross_entropy(output, labels)

                labels = labels.data.cpu().numpy()
                predic = torch.max(output.data, 1)[1].cpu().numpy()
                labels_all = np.append(labels_all, labels)
                predict_all = np.append(predict_all, predic)
        print("预测结果以及对应的预测得分：")
        for i in range(len(predict_all)):
            print("data %d score[ %f, %f] predice res %d" % (i, score_all[2 * i], score_all[2 * i + 1], predict_all[i]))
        # print(score_all)
        # print(predict_all)
        acc = metrics.accuracy_score(labels_all, predict_all)
        # print(acc)