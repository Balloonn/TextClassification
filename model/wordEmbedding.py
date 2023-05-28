import torch.nn as nn
import math


class WordEmbedding(nn.Module):
    def __init__(self, pretrained_vector, d_model):
        super(WordEmbedding, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding.from_pretrained(pretrained_vector)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
