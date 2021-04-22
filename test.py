import numpy as np
import torch, torchvision
from net.utils.graph import Graph


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:  # 构建邻接矩阵
        A[j, i] = 1  # 等同于A[j][i]
        A[i, j] = 1
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]  # 方矩阵乘法 d<0--对角矩阵;d>0--进行A的连乘
    temp = np.stack(transfer_mat)
    arrive_mat = (np.stack(transfer_mat) > 0)  # transfer_mat是list类型，需要将list堆叠成一个数组才能进行>操作
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return A


def normalize_digraph(A):  # 图卷积的预处理
    Dl = np.sum(A, 0)  # n*n矩阵求和变为n*1
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)  # 由每个点的度组成的对角矩阵
    AD = np.dot(A, Dn)
    return AD


num_node = 18
self_link = [(i, i) for i in range(num_node)]
neighbor_link = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12,
                                                            11),
                 (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
                 (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)]
edge = self_link + neighbor_link
A = get_hop_distance(num_node, edge)
# Graph.get_adjacency(A, 'spatial')
a = [[1, 2], [3, 4]]
b = [[5, 6], [6, 7]]
B = np.append(A, A)
C = [[1,2], [3, 4], [5, 6], [6, 7]]
C.append(a)
C = np.array(C)

#C_stack = np.stack(C)
print(C)
print('-----------------------------------------------------')
#print(C_stack.shape)

