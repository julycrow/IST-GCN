import torch
import torch.nn as nn


class MSTCN(nn.Module):
    def __init__(self,
                 out_channels,
                 kernel_size_a,
                 kernel_size_b,
                 kernel_size_c,
                 dropout,
                 stride=1
                 ):
        super().__init__()
        self.batchnorm2d = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv_a = nn.Conv2d(  # TCN，用的二维卷积，####这是一个1D卷积，卷积的维度是时间维度，他的卷积核一行多列
            out_channels,
            out_channels,
            kernel_size=(kernel_size_a, 1),  # 卷积核在第一位度大小为kernel_size[0]：卷积核的第一维即 9 帧，第二维度大小为1
            stride=(stride, 1),  # 步长
            padding=((kernel_size_a - 1) // 2, 0)  # padding
        )
        self.conv_b = nn.Conv2d(  # TCN，用的二维卷积，####这是一个1D卷积，卷积的维度是时间维度，他的卷积核一行多列
            out_channels,
            out_channels,
            kernel_size=(kernel_size_b, 1),  # 卷积核在第一位度大小为kernel_size[0]：卷积核的第一维即 9 帧，第二维度大小为1
            stride=(stride, 1),  # 步长
            padding=((kernel_size_b - 1) // 2, 0)  # padding
        )
        self.conv_c = nn.Conv2d(  # TCN，用的二维卷积，####这是一个1D卷积，卷积的维度是时间维度，他的卷积核一行多列
            out_channels,
            out_channels,
            kernel_size=(kernel_size_c, 1),  # 卷积核在第一位度大小为kernel_size[0]：卷积核的第一维即 9 帧，第二维度大小为1
            stride=(stride, 1),  # 步长
            padding=((kernel_size_c - 1) // 2, 0)  # padding
        )
        self.dropout = nn.Dropout(dropout, inplace=True)

    # 前向过程
    def forward(self, x, mstcn_importance):
        x = self.batchnorm2d(x)
        x = self.relu(x)
        # x_a = self.conv_a(x)
        x_b = self.conv_b(x)
        # x_c = self.conv_c(x)
        # x = torch.cat((x_a, x_b, x_c), 1)
        # x = x_a * mstcn_importance[0] + x_b * mstcn_importance[1] + x_c * mstcn_importance[2]
        x = x_b
        x = self.batchnorm2d(x)
        x = self.dropout(x)
        return x

