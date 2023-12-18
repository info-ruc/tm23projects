#!/usr/bin/env python3
from MyDataset import MyDataset
from MyNet import MyNet
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os

def accs(output, target):
    output = output.reshape(-1, vocab_size)
    target = target.flatten()
    a = output.topk(1).indices.flatten()
    b = target
    return a.eq(b).sum().item() / len(a)

def makewords(seqswords):
    seqswords = seqswords.split(exstr)
    hidden = None

    def wordItem(input_word):
        nonlocal hidden

        input_word_index = dataset.word2index[input_word]
        input_ = torch.Tensor([[input_word_index]]).long().to(device)
        output, hidden = model(input_, hidden)
        top_word_index = output[0].topk(1).indices.item()
        return dataset.index2word[top_word_index]

    result = [] 
    cur_word = exstr

    for i in range(seqs):
        if cur_word == exstr:
            result.append(cur_word)
            wordItem(cur_word)

            if len(seqswords) == 0:
                break

            for w in seqswords.pop(0):
                result.append(w)
                cur_word = wordItem(w)

        else:
            result.append(cur_word)
            cur_word = wordItem(cur_word)
    ress = "".join(result)
    ress = ress.strip(exstr)
    return ress


def training_action():
    for i, (input_, target) in enumerate(train_loader):
        model.train()

        input_, target = input_.to(device), target.to(device)

        output, _ = model(input_)
        loss = F.cross_entropy(output.reshape(-1, vocab_size), target.flatten())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = accs(output, target)

        print(
            "epoch=%d, batch=%d/%d, loss=%.4f, acc=%.4f"
            % (epoch, i, len(train_loader), loss.item(), acc)
        )

        if i % 3000 == 0:
            with open(middle_text,"a+",encoding='utf-8') as f:
                generate_data = makewords("秋#月")
                f.write(generate_data+"\n")
            f.close()

def evals():
    model.eval()
    lossval = 0
    acc = 0
    with torch.no_grad():
        for data in test_loader:
            input_, target = data[0].to(device), data[1].to(device)

            output, _ = model(input_)
            loss = F.cross_entropy(output.reshape(-1, vocab_size), target.flatten())

            acc += accs(output, target)

            lossval += loss.item()

    epoch_loss /= len(test_loader)
    acc /= len(test_loader)
    print(
        "epoch=%d, loss=%.4f, accuracy=%.4f"
        % (epoch, lossval, acc)
    )


def load_model(file):
    pt = torch.load(file)
    return pt

if __name__ == "__main__":

    debug = False
    embeding = 128
    hiddens = 1024
    lr = 0.001
    lstm_layers = 2
    batch_size = 35
    epochs = 10
    seqs = 7
    exstr = "#"
    middle_text = "./middle_text.txt"
    # 中间输出的测试值
    if os.path.exists(middle_text):
        os.remove(middle_text)

    device = torch.device("cuda:0")

    # 数据集
    dataset = MyDataset(seqs=seqs)

    data_length = len(dataset)
    lengths = [int(data_length - 1000), 1000]
    train_data, test_data = random_split(dataset, lengths)

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=True, num_workers=0
    )
    if debug:
        train_loader = [next(iter(train_loader))]
        test_loader = [next(iter(test_loader))]

    vocab_size = len(dataset.word2index)
    model = MyNet(
        vocab_size=vocab_size,
        embeding=embeding,
        hiddens=hiddens,
        lstm_layers=lstm_layers,
    )
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    epoch = 0
    # 循环训练
    while epoch < epochs:
        training_action()
        # evals()
        epoch+=1
    save = "./model.pkl"
    # 保存神经网络结果集
    try:
        torch.save(model, save)
    except:
        pass
