import torch
from torch import nn
import torch.nn.functional as F


# 构建基础的卷积模块，与Inception V2的基础模块比，增加了BN层
class BasicConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_padding=0,
                 t_kernel_size=1,
                 t_stride=1,
                 t_dilation=1,
                 bias=True
                 ):
        # super(BasicConv2d, self).__init__()
        super().__init__()

        # self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv = nn.Conv2d(  # 距离为1的卷积
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),  # 两个数分别作用在高和宽两个维度, 为1*1的卷积核,只能提取到自己的空间特征, 加入inception之后,可以提取到自己和周围节点的
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),  # 空洞卷积，默认为1（不采用），从卷积核上的一个参数到另一个参数需要走过的距离
            bias=bias)
        self.bn = nn.BatchNorm2d(out_channels * kernel_size)

    def forward(self, x):
        x = self.conv(x)
        return x


class Inception2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 ):
        # super(Inception2, self).__init__()
        super().__init__()
        self.kernel_size = kernel_size

        self.branch = BasicConv2d(in_channels, out_channels, kernel_size)

    # 前向过程
    def forward(self, x, A, A2, A3):
        assert A.size(0) == self.kernel_size
        x = self.branch(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)  # // -> 整除法

        x1 = torch.einsum('nkctv,kvw->nctw', [x, A])
        x2 = torch.einsum('nkctv,kvw->nctw', [x, A2])
        x3 = torch.einsum('nkctv,kvw->nctw', [x, A3])

        x = x1 + x2 + x3

        return x, A, A2, A3

