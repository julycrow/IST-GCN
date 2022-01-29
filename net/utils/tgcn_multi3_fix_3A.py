# The based unit of graph convolutional networks.

import torch
import torch.nn as nn


class ConvTemporalGraphical(nn.Module):
    r"""The basic module for applying a graph convolution.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``

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
    """
    需要用三个卷积核，都是一个1*1卷积核，但是输出通道是不同的，分别乘上三种不同距离邻接矩阵A，得到三种距离的关系，然后拼接在一起
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        self.kernel_size = kernel_size  # 卷积核数(向心离心静止), 不是卷积核大小
        self.conv = nn.Conv2d(  # 距离为1的卷积
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),  # 两个数分别作用在高和宽两个维度, 为1*1的卷积核,只能提取到自己的空间特征, 加入inception之后,可以提取到自己和周围节点的
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),  # 空洞卷积，默认为1（不采用），从卷积核上的一个参数到另一个参数需要走过的距离
            bias=bias)
        # self.conv2 = nn.Conv2d(  # 距离为2的卷积
        #     in_channels,
        #     out_channels * kernel_size,
        #     kernel_size=(2, 1),  # 两个数分别作用在高和宽两个维度, 为1*1的卷积核,只能提取到自己的空间特征, 加入inception之后,可以提取到自己和周围节点的
        #     padding=(1, 0),
        #     stride=(t_stride, 1),
        #     dilation=(t_dilation, 1),
        #     bias=bias)
        # self.conv3 = nn.Conv2d(  # 距离为3的卷积
        #     in_channels,
        #     out_channels * kernel_size,
        #     kernel_size=(3, 1),  # 两个数分别作用在高和宽两个维度, 为1*1的卷积核,只能提取到自己的空间特征, 加入inception之后,可以提取到自己和周围节点的
        #     padding=(2, 0),
        #     stride=(t_stride, 1),
        #     dilation=(t_dilation, 1),
        #     bias=bias)

    def forward(self, x, A, importance, importance2, importance3):
        assert A.size(0) == self.kernel_size
        # A.shape()=(3,25,25)
        x = self.conv(x)

        n, kc, t, v = x.size()  # (16,192,300,25)
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        # // -> 整数除法
        # x.shape()=(16,3,64,300,25)
        # 16=batchsize(8)*member(2)
        x1 = torch.einsum('nkctv,kvw->nctw', [x, A * importance])
        x2 = torch.einsum('nkctv,kvw->nctw', [x, A ** 2 * importance2])
        x3 = torch.einsum('nkctv,kvw->nctw', [x, A ** 3 * importance3])
        x = x1 + x2 + x3
        # 爱因斯坦简记法：做张量运算，'nkctv,kvw->nctw'为数组下标，其中隐含含义：对k,v进行求和  # x.shape()=(16,64,300,25)

        return x.contiguous(), A
