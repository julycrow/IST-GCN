from torch import nn
import torch


class TransFormerEncoder(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=54, nhead=6)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)

    def forward(self, x):
        x = self.encoder(x)
        return x


# src = torch.rand((64, 3, 150, 18, 2))
src = torch.rand((150, 32, 54))
print(type(src.size(1)))
c = 1
print(type(c))
# N, C, T, W = src.size()
# src = src.permute(2, 3, 0, 1).contiguous()
# temp = src.view(T * W, N, C)
# test = temp.view(T, V, M, N, C)
# src = temp
# print(src.shape)

# encoder_layer = nn.TransformerEncoderLayer(d_model=C, nhead=16)
# transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

# transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)
# out = transformer_encoder(src)
encoder_layer = nn.TransformerEncoderLayer(d_model=54, nhead=6)
encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
out = encoder(src)
print(out.shape)
