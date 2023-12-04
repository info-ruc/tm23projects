import argparse
import time
import math
import os

parser = argparse.ArgumentParser(description='TEST')
parser.add_argument('--lr', type=float, default=20, help='initial learning rate')
parser.add_argument('--epochs', type=int, default=40, help='upper epoch limit')
args = parser.parse_args()

def train():
    return 0

def evaluate():
    return 1.1

lr = args.lr
best_val_loss = None

for epoch in range(1, args.epochs+1):
    epoch_start_time = time.time()
    train()
    val_loss = evaluate()
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),val_loss, math.exp(val_loss)))
    print('-' * 89)
    if not best_val_loss or val_loss < best_val_loss:
        print("save model")
        best_val_loss = val_loss
    else:
        lr /= 4.0


