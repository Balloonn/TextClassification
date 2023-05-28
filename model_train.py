import torch
from numpy import argmax, vstack
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from utils import CSVDataset
from params import *
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from model.utils_model import clones, SublayerConnection, subsequent_mask, LayerNorm
from model.multiHeadAttention import MultiHeadAttention
from model.utils_model import FeedForward
from model.positionalEncoding import PositionalEncoding
from model.wordEmbedding import WordEmbedding
from model.encoder import Encoder, EncoderLayer
from model.generator import Generator
from get_pretrained_vector import pretrained_vector
from torch.autograd import Variable
from model.transformer import Transformer, MakeModel


class ModelTrainer(object):
    @staticmethod
    def make_std_mask(trg, pad):
        trg_mask = torch.Tensor((trg != pad)).int().unsqueeze(-2)  # batches * batch_size * seq_len
        trg_mask = trg_mask & Variable(subsequent_mask(trg.size(-1)).type_as(trg_mask.data))
        return trg_mask

    def train(self, model):
        train, test = CSVDataset(TRAIN_FILE_PATH), CSVDataset(TEST_FILE_PATH)
        train_dl = DataLoader(train, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
        optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
        for epoch in range(EPOCHS):
            for x_batch, y_batch in train_dl:
                y_batch = y_batch.long()
                optimizer.zero_grad()
                x_batch_mask = torch.Tensor((x_batch != PAD_NO)).int().unsqueeze(-2)
                y_pred = model(x_batch, x_batch_mask)
                loss = CrossEntropyLoss()(y_pred, y_batch)
                loss.backward()
                optimizer.step()

            print("Epoch: %d, loss: %.5f" % (epoch + 1, loss.item()))


if __name__ == '__main__':
    _model = MakeModel(clones, LayerNorm, SublayerConnection,
                       MultiHeadAttention, FeedForward,
                       PositionalEncoding,
                       Transformer,
                       Encoder, EncoderLayer,
                       WordEmbedding, Generator)
    model = _model.make_model(vocab_size=CHAR_NUM + 2,
                              pretrained_vector=pretrained_vector,
                              trg_vocab=5,
                              d_model=300,
                              d_ff=128,
                              n_heads=10,
                              n_layers=3,
                              dropout=0.1)
    num_params = sum(param.numel() for param in model.parameters())
    print("parameters: %d" % num_params)
    ModelTrainer().train(model)
    torch.save(model, 'sgns_cls.pth')
