import torch.nn as nn


class EncoderLayer(nn.Module):
    def __init__(self, clones, SublayerConnection, d_model, attn, feed_forward, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attn = attn
        self.feed_forward = feed_forward
        self.sublayer_connections = clones(SublayerConnection(d_model, dropout), 2)
        self.d_model = d_model

    def forward(self, x, src_mask):
        first_x = self.sublayer_connections[0](x, lambda x_attn: self.attn(x_attn, x_attn, x_attn, src_mask))
        return self.sublayer_connections[1](first_x, self.feed_forward)


class Encoder(nn.Module):
    def __init__(self, clones, LayerNorm, n, encoder_layer):
        super(Encoder, self).__init__()
        self.encoder_layers = clones(encoder_layer, n)
        self.layer_norm = LayerNorm(encoder_layer.d_model)

    def forward(self, x, src_mask):
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return self.layer_norm(x)
