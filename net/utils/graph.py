import numpy as np
import copy


class Graph():
    """ The Graph to model the skeletons extracted by the openpose

    Args:
        strategy (string): must be one of the follow candidates
        - uniform: Uniform Labeling
        - distance: Distance Partitioning
        - spatial: Spatial Configuration
        For more information, please refer to the section 'Partition Strategies'
            in our paper (https://arxiv.org/abs/1801.07455).

        layout (string): must be one of the follow candidates
        - openpose: Is consists of 18 joints. For more information, please
            refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose#output
        - ntu-rgb+d: Is consists of 25 joints. For more information, please
            refer to https://github.com/shahroudy/NTURGB-D

        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points

    """

    def __init__(self,
                 layout='openpose',
                 strategy='uniform',
                 max_hop=3,  # 最大距离
                 dilation=1,
                 kernel_size=3):
        self.max_hop = max_hop
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.get_edge(layout)
        self.adjacency_matrix, self.hop_dis = get_hop_distance(
            self.num_node, self.edge, self.spatial_symmetric, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A, self.A2, self.A3

    def get_edge(self, layout):
        if layout == 'openpose':
            self.num_node = 18
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12,
                                                                        11),
                             (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
                             (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)]
            self.edge = self_link + neighbor_link
            self.center = 1
        elif layout == 'openpose_gravity':
            self.num_node = 19
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11),
                             (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
                             (0, 1), (15, 0), (14, 0), (17, 15), (16, 14), (18, 0),
                             (18, 1), (18, 2), (18, 3), (18, 4), (18, 5), (18, 6),
                             (18, 7), (18, 8), (18, 9), (18, 10), (18, 11), (18, 12),
                             (18, 13), (18, 14), (18, 15), (18, 16), (18, 17)]
            self.edge = self_link + neighbor_link
            self.center = 1
        elif layout == 'openpose_symmetric':
            self.num_node = 18
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11),
                             (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
                             (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)]
            self.spatial_symmetric = [(14, 15), (16, 17), (2, 5), (3, 6), (4, 7), (8, 11), (9, 12), (10, 13)]
            self.edge = self_link + neighbor_link
            self.center = 1
        elif layout == 'ntu-rgb+d':
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                              (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                              (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                              (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                              (22, 23), (23, 8), (24, 25), (25, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 21 - 1
        elif layout == 'ntu_edge':
            self.num_node = 24
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (3, 2), (4, 3), (5, 2), (6, 5), (7, 6),
                              (8, 7), (9, 2), (10, 9), (11, 10), (12, 11),
                              (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                              (18, 17), (19, 18), (20, 19), (21, 22), (22, 8),
                              (23, 24), (24, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 2
        # elif layout=='customer settings'
        #     pass
        else:
            raise ValueError("Do Not Exist This Layout.")

    def get_adjacency(self, strategy):
        valid_hop = range(0, 1 + 1, self.dilation)  # 合法的距离值：0或1, valid_hop = [0, 1]
        # adjacency = np.zeros((self.num_node, self.num_node))
        # for hop in valid_hop:
        #     adjacency[self.hop_dis == hop] = 1  # 将0|1的位置置1,inf抛弃
        # normalize_adjacency = normalize_digraph(adjacency)  # 图卷积的预处理
        normalize_adjacency1 = get_norm(1, self.hop_dis, self.num_node, self.dilation)
        normalize_adjacency2 = get_norm(2, self.hop_dis, self.num_node, self.dilation)  # 距离为2的归一化距离矩阵
        normalize_adjacency3 = get_norm(3, self.hop_dis, self.num_node, self.dilation)
        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency1
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency1[self.hop_dis ==
                                                                 hop]
            self.A = A
        elif strategy == 'spatial':  # 建立关节节点分组
            A = []
            for hop in valid_hop:  # hop为0的时候找ij相等的点,即自身;hop为1的时候找相邻的点,分为近心和离心两类
                a_root = np.zeros((self.num_node, self.num_node))  # 前三组
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))

                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[i, self.center]:
                                a_root[j, i] = normalize_adjacency1[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[i, self.center]:
                                a_close[j, i] = normalize_adjacency1[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency1[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)

            A = np.stack(A)
            self.A = A
        elif strategy == 'spatial_gravity':  # 建立关节节点分组
            A = []
            for hop in valid_hop:  # hop为0的时候找ij相等的点,即自身;hop为1的时候找相邻的点,分为近心和离心两类
                a_root = np.zeros((self.num_node, self.num_node))  # 前三组
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))

                for i in range(self.num_node - 1):
                    for j in range(self.num_node - 1):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[i, self.center]:
                                a_root[j, i] = normalize_adjacency1[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[i, self.center]:
                                a_close[j, i] = normalize_adjacency1[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency1[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)

            a_gravity = np.zeros((self.num_node, self.num_node))  # 重心点分组
            for i in range(self.num_node):
                a_gravity[18, i] = normalize_adjacency1[18, i]
                a_gravity[i, 18] = normalize_adjacency1[i, 18]
            A.append(a_gravity)

            A = np.stack(A)
            self.A = A
        elif strategy == 'spatial_symmetric':  # 建立关节节点分组
            A = []
            for hop in valid_hop:  # hop为0的时候找ij相等的点,即自身;hop为1的时候找相邻的点,分为近心和离心两类
                a_root = np.zeros((self.num_node, self.num_node))  # 前三组
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))

                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[i, self.center]:
                                a_root[j, i] = normalize_adjacency1[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[i, self.center]:
                                a_close[j, i] = normalize_adjacency1[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency1[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)

            A = np.stack(A)
            A2 = get_A(A, normalize_adjacency2, self.adjacency_matrix, self.num_node, self.kernel_size)
            A3 = get_A(A2, normalize_adjacency3, self.adjacency_matrix, self.num_node, self.kernel_size)
            A = every_symmetric(A, normalize_adjacency2, self.num_node, self.spatial_symmetric)
            A2 = every_symmetric(A2, normalize_adjacency2, self.num_node, self.spatial_symmetric)
            A3 = every_symmetric(A3, normalize_adjacency3, self.num_node, self.spatial_symmetric)
            self.A = A
            self.A2 = A2
            self.A3 = A3
        else:
            raise ValueError("Do Not Exist This Strategy")


def get_hop_distance(num_node, edge, spatial_symmetric, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:  # 构建邻接矩阵
        A[j, i] = 1  # 等同于A[j][i]
        A[i, j] = 1
    adjacency_matrix = copy.deepcopy(A)  # 没有对称点的邻接矩阵，用于后面分组
    for i, j in spatial_symmetric:
        A[j, i] = 1
        A[i, j] = 1
    '''
    A=[[1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 1. 0. 0.]
     [1. 1. 1. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 1. 1. 1. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 1. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 1. 0. 0. 0. 1. 1. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 1. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 1. 0. 0. 0. 0. 0. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 1. 1. 1. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 1. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 1. 1. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 1. 1. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 1. 0. 0. 0. 0.]
     [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 1. 0.]
     [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 1.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 1. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 1.]]
    '''

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]  # 方矩阵乘法
    arrive_mat = (np.stack(transfer_mat) > 0)  # transfer_mat是list类型，需要将list堆叠成一个数组才能进行>操作
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    '''
    hop_dis=
    [[ 0.  1. inf inf inf inf inf inf inf inf inf inf inf inf  1.  1. inf inf]
     [ 1.  0.  1. inf inf  1. inf inf inf inf inf inf inf inf inf inf inf inf]
     [inf  1.  0.  1. inf inf inf inf  1. inf inf inf inf inf inf inf inf inf]
     [inf inf  1.  0.  1. inf inf inf inf inf inf inf inf inf inf inf inf inf]
     [inf inf inf  1.  0. inf inf inf inf inf inf inf inf inf inf inf inf inf]
     [inf  1. inf inf inf  0.  1. inf inf inf inf  1. inf inf inf inf inf inf]
     [inf inf inf inf inf  1.  0.  1. inf inf inf inf inf inf inf inf inf inf]
     [inf inf inf inf inf inf  1.  0. inf inf inf inf inf inf inf inf inf inf]
     [inf inf  1. inf inf inf inf inf  0.  1. inf inf inf inf inf inf inf inf]
     [inf inf inf inf inf inf inf inf  1.  0.  1. inf inf inf inf inf inf inf]
     [inf inf inf inf inf inf inf inf inf  1.  0. inf inf inf inf inf inf inf]
     [inf inf inf inf inf  1. inf inf inf inf inf  0.  1. inf inf inf inf inf]
     [inf inf inf inf inf inf inf inf inf inf inf  1.  0.  1. inf inf inf inf]
     [inf inf inf inf inf inf inf inf inf inf inf inf  1.  0. inf inf inf inf]
     [ 1. inf inf inf inf inf inf inf inf inf inf inf inf inf  0. inf  1. inf]
     [ 1. inf inf inf inf inf inf inf inf inf inf inf inf inf inf  0. inf  1.]
     [inf inf inf inf inf inf inf inf inf inf inf inf inf inf  1. inf  0. inf]
     [inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf  1. inf  0.]]
    '''
    return adjacency_matrix, hop_dis  # hop_dis 各节点之间的路径长度--只有0,1,2,其他为inf（含有对称点）
    # adjacency_matrix 没有对称点的邻接矩阵，用于后面分组


def normalize_digraph(A):  # 有向图卷积的预处理
    Dl = np.sum(A, 0)  # n*n矩阵求和变为n*1
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)  # 由每个点的度组成的对角矩阵
    AD = np.dot(A, Dn)  # 点乘
    return AD


'''
AD
0.25,0.25,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.33,0.33,0.00,0.00
0.25,0.25,0.25,0.00,0.00,0.25,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
0.00,0.25,0.25,0.33,0.00,0.00,0.00,0.00,0.33,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
0.00,0.00,0.25,0.33,0.50,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
0.00,0.00,0.00,0.33,0.50,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
0.00,0.25,0.00,0.00,0.00,0.25,0.33,0.00,0.00,0.00,0.00,0.33,0.00,0.00,0.00,0.00,0.00,0.00
0.00,0.00,0.00,0.00,0.00,0.25,0.33,0.50,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
0.00,0.00,0.00,0.00,0.00,0.00,0.33,0.50,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
0.00,0.00,0.25,0.00,0.00,0.00,0.00,0.00,0.33,0.33,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.33,0.33,0.50,0.00,0.00,0.00,0.00,0.00,0.00,0.00
0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.33,0.50,0.00,0.00,0.00,0.00,0.00,0.00,0.00
0.00,0.00,0.00,0.00,0.00,0.25,0.00,0.00,0.00,0.00,0.00,0.33,0.33,0.00,0.00,0.00,0.00,0.00
0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.33,0.33,0.50,0.00,0.00,0.00,0.00
0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.33,0.50,0.00,0.00,0.00,0.00
0.25,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.33,0.00,0.50,0.00
0.25,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.33,0.00,0.50
0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.33,0.00,0.50,0.00
0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.33,0.00,0.50
'''


def normalize_undigraph(A):  # 无向图
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD


def get_norm(max_hop, hop_dis, num_node, dilation):
    valid_hop = range(0, max_hop + 1, dilation)  # 合法的距离值：0或1, valid_hop = [0, 1]
    adjacency = np.zeros((num_node, num_node))
    for hop in valid_hop:
        adjacency[hop_dis == hop] = 1  # 将0|1的位置置1,inf抛弃
    normalize_adjacency = normalize_digraph(adjacency)  # 图卷积的预处理
    return normalize_adjacency


def add_one_distance(adjacency_matrix, A, normalize_adjacency, num_node, kernel_size):  # 对A增加1个距离，得到更远距离的归一化矩阵
    res = copy.deepcopy(A)
    for kernel in range(1, kernel_size):  # 1-离心, 2-向心
        for i in range(num_node):
            for j in range(num_node):
                if res[kernel][j, i] != 0:
                    res[kernel][j, i] = normalize_adjacency[j, i]
                    for k in range(num_node):
                        if adjacency_matrix[j][k] == 1 and res[1][k, i] == 0 and k != i:
                            res[kernel][k, i] = normalize_adjacency[k, i]
    return res


def get_A(A, normalize_adjacency, adjacency_matrix, num_node, kernel_size):
    # res = np.zeros(shape=[max_hop, 3, num_node, num_node])
    temp = copy.deepcopy(A)
    temp = add_one_distance(adjacency_matrix, temp, normalize_adjacency, num_node, kernel_size)
    return temp


def every_symmetric(A, normalize_adjacency, num_node, spatial_symmetric):  # 加入对称点
    symmetric = np.zeros((num_node, num_node))
    for i, j in spatial_symmetric:
        symmetric[i, j] = normalize_adjacency[i, j]
    symmetric = np.expand_dims(symmetric, axis=0)
    # np.stack([A, symmetric], axis=0)
    A = np.append(A, symmetric, axis=0)
    # np.concatenate((A, symmetric), axis=0)
    return A
