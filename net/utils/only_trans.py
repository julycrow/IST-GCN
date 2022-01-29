import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from net.utils.tgcn import ConvTemporalGraphical
from net.utils.graph import Graph
from net.utils.encoder import TransFormerEncoder
import torchsnooper


class Model(nn.Module):
    r"""Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self, num_class, in_channels, graph_args):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)  # 注册变量,A是tensor变量。在之后的调用只用self.A_即可调用，寄存器变量访问快。

        # build networks
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))  # 批量归一化

        # fcn for prediction 全连接
        self.fcn = nn.Conv2d(54, num_class, kernel_size=1)
        self.fc1 = nn.Linear(3456, 4)

    def forward(self, x, num_class):
        # data normalization
        N, C, T, V, M = x.size()  # 整个网络的输入是一个(N = batch_size,C = 3,T = 150,V = 18,M = 2)的tensor
        x = x.permute(0, 4, 3, 1, 2).contiguous()  # 把tensor变成在内存中连续分布的形式,这样才可以view
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(4, 0, 1, 3, 2).contiguous()
        x = x.view(T, N * M, C * V)  # 对应 S N E

        # forward
        encoder_layer = nn.TransformerEncoderLayer(d_model=54, nhead=6).cuda()
        encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        x = encoder(x)
        x = x.view(x.size(0), -1)
        # global pooling

        # prediction
        print(type(x.size(1)))
        # x = nn.Linear(x, in_features=x.size(1), num_class)
        x = self.fc1(x)
        x = x.view(x.size(0), -1)

        return x
