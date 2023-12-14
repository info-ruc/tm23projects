import torch
from torch.optim import RMSprop, Adam
from torch.nn import CrossEntropyLoss, Softmax
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

from numpy import vstack, argmax
from sklearn.metrics import accuracy_score

from net.model import TextClassifier
from utils.text_featuring import CSVDataset
from config.params import TRAIN_BATCH_SIZE, TEST_BATCH_SIZE, LEARNING_RATE, EPOCHS, TRAIN_FILE_PATH, TEST_FILE_PATH, MODEL_PATH

class ModelTrainer(object):
    def evaluate_model(self, test_dl, model):
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

    def train(self, model):
        # calculate split
        train, test = CSVDataset(TRAIN_FILE_PATH), CSVDataset(TEST_FILE_PATH)
        # prepare data loaders
        train_dl = DataLoader(train, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
        test_dl = DataLoader(test, batch_size=TEST_BATCH_SIZE)

        # Define optimizer
        optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
        # Starts training phase
        writer = SummaryWriter('logs')

        loss_avg = 0
        for epoch in range(EPOCHS):
            loss_avg = 0
            # Starts batch training
            model.train()
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
                loss_avg += loss.item()

            loss_avg /= len(train_dl)

            # Evaluation
            model.eval()
            test_accuracy = self.evaluate_model(test_dl, model)
            print("Epoch: %d, loss: %.5f, Test accuracy: %.5f" % (epoch+1, loss_avg, test_accuracy))

            writer.add_scalar('loss', loss_avg, epoch)
            writer.add_scalar('acc', test_accuracy, epoch)


if __name__ == '__main__':
    model = TextClassifier(nhead=10,             # number of heads in the multi-head-attention models
                           dim_feedforward=128,  # dimension of the feedforward network model in nn.TransformerEncoder
                           num_layers=1,
                           dropout=0.1,
                           classifier_dropout=0.1)
    ModelTrainer().train(model)
    torch.save(model, MODEL_PATH)
