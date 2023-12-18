import torch
from numpy import vstack, argmax
from sklearn.metrics import accuracy_score
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from common import TRAIN_BATCH_SIZE, TEST_BATCH_SIZE, LEARNING_RATE, EPOCHS, TRAIN_FILE_PATH, TEST_FILE_PATH, MODEL_FILE_PATH
from file_operation import CSVDataset
from transformer_model import TextClassifier

# 模型训练
class ModelTrainer(object):
    # 模型评估
    @staticmethod
    def evaluate_model(test_dl, model):
        predictions, actuals = [], []
        for i, (inputs, targets) in enumerate(test_dl):
            # evaluate the model on the test set
            yhat = model(inputs)
            # retrieve numpy array
            yhat = yhat.detach().numpy()
            actual = targets.numpy()
            # convert to class labels
            yhat = argmax(yhat, axis=1)
            # reshape for stacking
            actual = actual.reshape((len(actual), 1))
            yhat = yhat.reshape((len(yhat), 1))
            # store
            predictions.append(yhat)
            actuals.append(actual)
        predictions, actuals = vstack(predictions), vstack(actuals)
        # calculate accuracy
        acc = accuracy_score(actuals, predictions)
        return acc

    # 模型训练，评估和度量计算
    def train(self, model):
        # calculate split
        train, test = CSVDataset(TRAIN_FILE_PATH), CSVDataset(TEST_FILE_PATH)
        # prepare data loaders
        train_dl = DataLoader(train, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
        test_dl = DataLoader(test, batch_size=TEST_BATCH_SIZE)

        # Define optimizer
        optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
        # Starts training phase
        for epoch in range(EPOCHS):
            # Starts batch training
            for x_batch, y_batch in train_dl:
                y_batch = y_batch.long()
                # Clean gradients
                optimizer.zero_grad()
                # Feed the model
                y_pred = model(x_batch)
                # Loss calculation
                loss = CrossEntropyLoss()(y_pred, y_batch)
                # Gradients calculation
                loss.backward()
                # Gradients update
                optimizer.step()

            # Evaluation
            test_accuracy = self.evaluate_model(test_dl, model)
            print("训练轮次: %d, 损失率: %.5f, 准确率: %.5f" % (epoch+1, loss.item(), test_accuracy))


if __name__ == '__main__':
    model = TextClassifier(nhead=10,             # number of heads in the multi-head-attention models
                           dim_feedforward=2048,  # dimension of the feedforward network model in nn.TransformerEncoder
                           nlayers=2,
                           dropout=0.1)
    print()
    # 统计参数量
    num_params = sum(param.numel() for param in model.parameters())
    print(f"模型参数量: {num_params}")
    # 训练模型
    ModelTrainer().train(model)
    # 保存模型
    torch.save(model, MODEL_FILE_PATH)
    # torch.save(model, 'model/cnews_text_cl_cn1.pth')
