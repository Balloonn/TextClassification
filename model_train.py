import torch
from numpy import argmax, vstack
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from utils import CSVDataset
from params import *
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from model.transformer import make_model
from get_pretrained_vector import pretrained_vector


class ModelTrainer(object):
    @staticmethod
    def evaluate_model(test_dl, model):
        predictions, labels = [], []
        for i, (inputs, targets) in enumerate(test_dl):
            yhat = model(inputs)
            yhat = yhat.detach().numpy()
            yhat = argmax(yhat, axis=1)
            label = targets.numpy()

            yhat = yhat.reshape(len(yhat), 1)
            label = label.reshape(len(label), 1)

            predictions.append(yhat)
            labels.append(label)

            predictions, labels = vstack(predictions), vstack(labels)

            acc = accuracy_score(labels, predictions)
            return acc

    def train(self, model):
        train, test = CSVDataset(TRAIN_FILE_PATH), CSVDataset(TEST_FILE_PATH)
        train_dl = DataLoader(train, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
        test_dl = DataLoader(test, batch_size=TEST_BATCH_SIZE)
        optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

        for epoch in range(EPOCHS):
            for x_batch, y_batch in train_dl:
                y_batch = y_batch.long()
                optimizer.zero_grad()
                y_pred = model(x_batch)
                loss = CrossEntropyLoss()(y_pred, y_batch)
                loss.backward()
                optimizer.step()

        test_accuracy = self.evaluate_model(test_dl, model)
        print("Epoch: %d, loss: %.5f, Test accuracy: %.5f" % (epoch + 1, loss.item(), test_accuracy))


if __name__ == '__main__':
    model = make_model(pretrained_vector=pretrained_vector,
                       trg_vocab=5,
                       d_model=300,
                       d_ff=2048,
                       n_heads=6,
                       n_layers=6,
                       dropout=0.1)
    num_params = sum(param.numel() for param in model.parameters())
    print("parameters: %d" % num_params)
    ModelTrainer().train(model)
    torch.save(model, 'sgns_cls.pth')
