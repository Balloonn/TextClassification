from multiHeadAttention import MultiHeadAttention
from utils_model import FeedForward
from positionalEncoding import PositionalEncoding
from wordEmbedding import WordEmbedding
from encoder import Encoder, EncoderLayer
from decoder import Decoder, DecoderLayer
from generator import Generator
import copy as c
import torch.nn as nn


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, trg_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.generator = generator

    def encode(self, src, src_mask):
        x = self.src_embed(src)
        x = self.encoder(x, src_mask)
        return x

    def decode(self, trg, memory, src_mask, trg_mask):
        x = self.trg_embed(trg)
        x = self.decoder(x, memory, src_mask, trg_mask)
        return x

    def forward(self, src, trg, src_mask, trg_mask):
        encoder_outputs = self.encode(src, src_mask)
        decoder_outputs = self.decode(trg, encoder_outputs, src_mask, trg_mask)
        return decoder_outputs


def make_model(pretrained_vector, trg_vocab, d_model=512, d_ff=2048, n_heads=8, n_layers=6, dropout=0.1):
    attn = MultiHeadAttention(n_heads, d_model, dropout)
    feed_forward = FeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(n_layers, EncoderLayer(d_model, c.deepcopy(attn), c.deepcopy(feed_forward), dropout)),
        Decoder(n_layers, DecoderLayer(d_model, c.deepcopy(attn), c.deepcopy(attn), c.deepcopy(feed_forward), dropout)),
        nn.Sequential(WordEmbedding(pretrained_vector, d_model), c.deepcopy(position)),
        nn.Sequential(WordEmbedding(pretrained_vector, d_model), c.deepcopy(position)),
        Generator(d_model, trg_vocab)
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model
