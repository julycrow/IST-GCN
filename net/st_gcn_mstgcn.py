import torch
import torch.nn as nn
import torch.nn.functional as F

from net.utils.graph import Graph
from net.utils.inceptionv2_gcn import Inception2
from net.utils.ms_tcn import MSTCN

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

    def __init__(self, in_channels, num_class, graph_args,
                 edge_importance_weighting, **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_args)
        # if self.arg.model_args.graph_args.strategy == 'spatial_3' or self.arg.model_args.graph_args.strategy == 'spatial_3_sym':
        A2 = torch.tensor(self.graph.A2, dtype=torch.float32, requires_grad=False)
        A3 = torch.tensor(self.graph.A3, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A2', A2)
        self.register_buffer('A3', A3)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)  # 注册变量,A是tensor变量。在之后的调用只用self.A_即可调用，寄存器变量访问快。

        # build networks
        spatial_kernel_size = A.size(0)  # A.shape() = (3,18,18)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)  # (9, 3)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))  # 批量归一化
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((
            # st_gcn(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
            # st_gcn(64, 64, kernel_size, 1, **kwargs),
            # st_gcn(64, 64, kernel_size, 1, **kwargs),
            # st_gcn(64, 64, kernel_size, 1, **kwargs),
            # st_gcn(64, 128, kernel_size, 2, **kwargs),
            # st_gcn(128, 128, kernel_size, 1, **kwargs),
            # st_gcn(128, 128, kernel_size, 1, **kwargs),
            # st_gcn(128, 256, kernel_size, 2, **kwargs),
            # st_gcn(256, 256, kernel_size, 1, **kwargs),
            # st_gcn(256, 256, kernel_size, 1, **kwargs),
            st_gcn(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
            # st_gcn(32, 32, kernel_size, 1, **kwargs),
            # st_gcn(32, 64, kernel_size, 2, **kwargs),
            # st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),

            st_gcn(64, 128, kernel_size, 2, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
            # st_gcn(128, 128, kernel_size, 1, **kwargs),

            st_gcn(128, 256, kernel_size, 2, **kwargs),
            # st_gcn(256, 256, kernel_size, 1, **kwargs),
            st_gcn(256, 256, kernel_size, 1, ** kwargs)

        ))

        # initialize parameters for edge importance weighting   初始化边重要性权重的参数(简单的注意力机制)
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))  # initial with one
                for i in self.st_gcn_networks
            ])
            self.edge_importance2 = nn.ParameterList([
                nn.Parameter(torch.ones(self.A2.size()))  # initial with one
                for i in self.st_gcn_networks
            ])
            self.edge_importance3 = nn.ParameterList([
                nn.Parameter(torch.ones(self.A3.size()))  # initial with one
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)
            self.edge_importance2 = [1] * len(self.st_gcn_networks)
            self.edge_importance3 = [1] * len(self.st_gcn_networks)

        # fcn for prediction 全连接
        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)
        # self.transformerencoder = TransFormerEncoder(256, 256)

    def forward(self, x):

        # data normalization
        N, C, T, V, M = x.size()  # 整个网络的输入是一个(N = batch_size,C = 3,T = 150,V = 18,M = 2)的tensor
        x = x.permute(0, 4, 3, 1, 2).contiguous()  # 把tensor变成在内存中连续分布的形式,这样才可以view
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)  # 在进行2维卷积(n,c,h,w)的时候需要将 N 与 M 合并起来形成(N * M, C, T, V)
        # 换成这样的格式就可以与2维卷积完全类比起来。CNN中核的两维对应的是(h,w)，而st-gcn的核对应的是(T,V)

        # forward
        for gcn, importance, importance2, importance3 in zip(self.st_gcn_networks, self.edge_importance,
                                                             self.edge_importance2,
                                                             self.edge_importance3):  # edge_importance与st-gcn层一一对应,
            # gcn实际上代表的是st-gcn的一层
            x, _, _2, _3 = gcn(x, self.A * importance, self.A2 * importance2,
                               self.A3 * importance3)  # 注意在forward传入的A并不是单纯的self.A,而是self.A * importance
        # for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):  # 公用一个权重,
        #     # gcn实际上代表的是st-gcn的一层
        #     x, _, _2, _3 = gcn(x, self.A * importance, self.A2 * importance,
        #                        self.A3 * importance)  # 注意在forward传入的A并不是单纯的self.A,而是self.A * importance
        # x = self.transformerencoder(x)
        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)

        # prediction
        x = self.fcn(x)
        x = x.view(x.size(0), -1)

        return x

    def extract_feature(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()  # permute将tensor的维度换位，contiguous将tensor变成在内存中连续分布的形式
        x = x.view(N * M, V * C, T)  # 维度转换
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        _, c, t, v = x.size()
        feature = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)

        # prediction
        x = self.fcn(x)
        output = x.view(N, M, -1, t, v).permute(0, 2, 3, 4, 1)

        return output, feature


class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2  # 断言：当表达式为真时，程序继续往下执行；当表达式为假时，抛出AssertionError错误，并将参数输出
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        # self.gcn = ConvTemporalGraphical(in_channels, out_channels,
        #                                  kernel_size[1])  # 使用卷积核的第二维即 3 组
        self.gcn = Inception2(in_channels, out_channels,
                              kernel_size[1])  # 使用卷积核的第二维即 3 组
        # self.tcn = nn.Sequential(  # 一个有序的容器，神经网络模块将按照在传入构造器的顺序依次被添加到计算图中执行
        #     nn.BatchNorm2d(out_channels),  # 在卷积神经网络的卷积层之后总会添加BatchNorm2d进行数据的归一化处理，这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(  # TCN，用的二维卷积，####这是一个1D卷积，卷积的维度是时间维度，他的卷积核一行多列
        #         out_channels,
        #         out_channels,
        #         (kernel_size[0], 1),  # 卷积核在第一位度大小为kernel_size[0]：卷积核的第一维即 9 帧，第二维度大小为1
        #         (stride, 1),  # 步长
        #         padding,  # 填充
        #     ),
        #     nn.BatchNorm2d(out_channels),
        #     nn.Dropout(dropout, inplace=True),  # inplace=True->在通过relu()计算时的得到的新值不会占用新的空间而是直接覆盖原来的值，为False则相反
        # )
        self.tcn = MSTCN(out_channels, 3, 9, 15, dropout, stride)
        if not residual:  # 每一个st-gcn层都用residual残差模块来改进
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),  # 当通道数要增加时，使用1x1conv来进行通道的翻倍
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A, A2, A3):

        res = self.residual(x)
        x, A, A2, A3 = self.gcn(x, A, A2, A3)
        x = self.tcn(x) + res
        # x = self.transformerencoder(x) + res
        return self.relu(x), A, A2, A3
