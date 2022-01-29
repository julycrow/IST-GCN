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
        self.bn = nn.BatchNorm2d(out_channels * kernel_size, eps=0.001)  # 分母中添加的一个值，目的是为了计算的稳定性，默认为：1e-5

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class Inception2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size
                 ):
        # super(Inception2, self).__init__()
        super().__init__()
        self.kernel_size = kernel_size
        # 具体对应如Inception v2网络结构图（上图）
        # 对应1x1卷积分支
        self.branch1 = BasicConv2d(in_channels, out_channels // 8, kernel_size)
        # 对应1x1卷积与3x3卷积分支
        self.branch2A = BasicConv2d(in_channels, out_channels // 4, kernel_size)
        self.branch2B = BasicConv2d(out_channels // 4, out_channels // 8 * 3, kernel_size)
        # 对应1x1卷积，3x3卷积，3x3卷积
        self.branch3A = BasicConv2d(in_channels, out_channels // 4, kernel_size)
        # self.branch3B = BasicConv2d(out_channels // 4, out_channels // 4, kernel_size),
        self.branch3C = BasicConv2d(out_channels // 4, out_channels // 8, kernel_size)
        # 对应3x3平均池化和1x1卷积
        self.branch4A = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        self.branch4B = BasicConv2d(in_channels, out_channels // 8 * 3, kernel_size)

    # 前向过程
    def forward(self, x, A, A2, A3):
        assert A.size(0) == self.kernel_size
        x0 = self.branch1(x)
        x0 = self.multiply_Distance_matrix(x0, A)

        x1a = self.branch2A(x)
        x1a = self.multiply_Distance_matrix(x1a, A)
        x1b = self.branch2B(x1a)
        x1 = self.multiply_Distance_matrix(x1b, A2)

        x2a = self.branch3A(x)
        x2a = self.multiply_Distance_matrix(x2a, A)
        x2c = self.branch3C(x2a)
        x2 = self.multiply_Distance_matrix(x2c, A3)

        x3a = self.branch4A(x)
        x3a = self.branch4B(x3a)
        x3 = self.multiply_Distance_matrix(x3a, A)

        out = torch.cat((x0, x1, x2, x3), 1)
        return out, A, A2, A3

    def multiply_Distance_matrix(self, x, A):
        n, kc, t, v = x.size()
        x0 = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)  # // -> 整除法
        x0 = torch.einsum('nkctv,kvw->nctw', [x0, A])  # 爱因斯坦简记法：做张量运算，'nkctv,kvw->nctw'为数组下标，其中隐含含义：对k,v进行求和
        return x0
