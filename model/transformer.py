import copy as c
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(self, encoder, src_embed, generator):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.src_embed = src_embed
        self.generator = generator

    def encode(self, src, src_mask):
        x = self.src_embed(src)
        x = self.encoder(x, src_mask)
        return x

    def forward(self, src, src_mask):
        encoder_outputs = self.encode(src, src_mask)
        encoder_outputs = encoder_outputs.mean(dim=1)
        output = self.generator(encoder_outputs)
        return output


class MakeModel:
    def __init__(self, clones, LayerNorm, SublayerConnection,
                 multi_head_attention, feed_forward,
                 positional_encoding,
                 transformer,
                 encoder, encoder_layer,
                 word_embedding, generator):
        self.clones = clones
        self.LayerNorm = LayerNorm
        self.SublayerConnection = SublayerConnection
        self.multi_head_attention = multi_head_attention
        self.feed_forward = feed_forward
        self.positional_encoding = positional_encoding
        self.transformer = transformer
        self.encoder = encoder
        self.encoder_layer = encoder_layer
        self.word_embedding = word_embedding
        self.generator = generator

    def make_model(self, vocab_size, pretrained_vector,
                   trg_vocab, d_model=512,
                   d_ff=2048, n_heads=8,
                   n_layers=6, dropout=0.1):
        attn = self.multi_head_attention(n_heads, d_model, dropout)
        feed_forward = self.feed_forward(d_model, d_ff, dropout)
        position = self.positional_encoding(d_model, dropout, vocab_size)
        model = self.transformer(
            self.encoder(self.clones, self.LayerNorm, n_layers,
                         self.encoder_layer(self.clones, self.SublayerConnection,
                                            d_model, c.deepcopy(attn),
                                            c.deepcopy(feed_forward), dropout)),
            nn.Sequential(self.word_embedding(d_model, pretrained_vector, None), c.deepcopy(position)),
            self.generator(d_model, trg_vocab)
        )

        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)
        return model
