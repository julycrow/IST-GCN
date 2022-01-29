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
        self.bn = nn.BatchNorm2d(out_channels * kernel_size)  # 分母中添加的一个值，目的是为了计算的稳定性，默认为：1e-5

    def forward(self, x):
        x = self.conv(x)
        # x = self.bn(x)
        return x


class Inception2(nn.Module):
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
        # super(Inception2, self).__init__()
        super().__init__()
        self.kernel_size = kernel_size
        # self.conv = nn.Conv2d(  # 距离为1的卷积
        #     in_channels,
        #     out_channels * kernel_size,
        #     kernel_size=(t_kernel_size, 1),  # 两个数分别作用在高和宽两个维度, 为1*1的卷积核,只能提取到自己的空间特征, 加入inception之后,可以提取到自己和周围节点的
        #     padding=(t_padding, 0),
        #     stride=(t_stride, 1),
        #     dilation=(t_dilation, 1),  # 空洞卷积，默认为1（不采用），从卷积核上的一个参数到另一个参数需要走过的距离
        #     bias=bias)

        self.branch = BasicConv2d(in_channels, out_channels, kernel_size)

    # 前向过程
    def forward(self, x, A, A2, A3):
        assert A.size(0) == self.kernel_size
        # x0 = self.conv(x)
        x0 = self.branch(x)

        x1 = self.multiply_Distance_matrix(x0, A)
        x2 = self.multiply_Distance_matrix(x0, A2)
        x3 = self.multiply_Distance_matrix(x0, A3)

        # x1 = self.avg_x(x1, 2)
        # x2 = self.avg_x(x2, 4)
        # x3 = self.avg_x(x3, 4)

        # x1 = self.avg_pool(x1, (1, 2))
        # x2 = self.avg_pool(x2, (1, 4))
        # x3 = self.avg_pool(x3, (1, 4))
        out = x1 + x2 + x3
        # out = torch.cat((x1, x2, x3), 1)
        # out = x1 + x2
        return out, A, A2, A3

    def multiply_Distance_matrix(self, x, A):
        n, kc, t, v = x.size()
        x0 = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)  # // -> 整除法
        x0 = torch.einsum('nkctv,kvw->nctw', [x0, A])  # 爱因斯坦简记法：做张量运算，'nkctv,kvw->nctw'为数组下标，其中隐含含义：对k,v进行求和
        return x0.contiguous()
    #
    # def avg_pool(self, x, kernel_size):  # 平均池化
    #     x = x.permute(0, 2, 3, 1)
    #     x = F.avg_pool2d(x, kernel_size=kernel_size)
    #     x = x.permute(0, 3, 1, 2)
    #     return x
    #
    # def avg_x(self, x, k):
    #     n, c, t, v = x.size()
    #     x = x.view(n, c // k, k, t, v)
    #     x = x.mean(dim=2)
    #     return x
