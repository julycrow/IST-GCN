import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, residual=False, variant=False):
        super(GraphConvolution, self).__init__()
        self.variant = variant
        if self.variant:
            self.in_features = 2 * in_features
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        # self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, h0, lamda, alpha, l):
        theta = math.log(lamda / l + 1)
        # n, kc, t, v = input.size()
        # hi = input.view(n, 3, kc // 3, t, v)  # // -> 整除法
        # hi = torch.einsum('nkctv,kvw->nctw', [hi, adj])  # 爱因斯坦简记法：做张量运算，'nkctv,kvw->nctw'为数组下标，其中隐含含义：对k,v进行求和
        # hi = hi.contiguous()
        # hi = torch.spmm(input, adj)
        h1 = torch.matmul(input, adj[0])
        h2 = torch.matmul(input, adj[1])
        h3 = torch.matmul(input, adj[2])
        hi = h1 + h2 + h3
        if self.variant:
            support = torch.cat([hi, h0], 1)
            r = (1 - alpha) * hi + alpha * h0
        else:
            support = (1 - alpha) * hi + alpha * h0  # initial residual
            r = support
        n, kc, t, v = support.size()
        support = support.view(n, t, v, kc)
        output = torch.matmul(support, self.weight)
        output = output.view(n, kc, t, v)
        # output = theta * output.permute(0, 3, 1, 2) + (1 - theta) * r  # identity mapping
        output = theta * output + (1 - theta) * r
        if self.residual:
            output = output + input
        return output
