import torch.nn as nn
import math


class WordEmbedding(nn.Module):
    def __init__(self, d_model, pretrained_vector=None, vocab_size=None):
        super(WordEmbedding, self).__init__()
        self.d_model = d_model
        if pretrained_vector is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_vector, freeze=False, padding_idx=0)
        if vocab_size is not None:
            self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
