import torch
import torch.nn as nn
from net.utils.graph import Graph


class TransFormerEncoder(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,

                 stride=1
                 ):
        super().__init__()
        # self.graph = Graph(**graph_args)
        # A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        # self.register_buffer('A', A)  # 注册变量,A是tensor变量。在之后的调用只用self.A_即可调用，寄存器变量访问快。
        self.stride = stride
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=out_channels * 18, nhead=6)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        # if in_channels != out_channels:
        #     self.stride = 2
        # else:
        #     self.stride = 1

        # self.data_bn = nn.BatchNorm1d(out_channels * A.size(1))
        self.residual = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=(stride, 1)),  # 当通道数要增加时，使用1x1conv来进行通道的翻倍
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        N, C, T, W = x.size()
        # x = x.permute(2, 3, 0, 1).contiguous()
        # x = x.view(T * W, N, C)
        # x = x.permute(3, 0, 1, 2).contiguous()
        x = x.permute(2, 0, 1, 3).contiguous()

        x = x.view(T, N * C, W)
        # x = self.data_bn(x)
        x, _ = self.encoder(x)
        # x = x.view(T, W, N, C)
        x = x.view(T, N, C, W)
        # x = x.permute(2, 3, 0, 1)
        x = x.permute(1, 2, 0, 3).contiguous()
        # if self.stride == 1:
        #     # 每两个取均值
        #     # a, b = x.split(T/2, dim=2)  # 75->38会报错
        #     x = self.residual(x)
        return x
