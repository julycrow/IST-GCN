#!/usr/bin/env python
# pylint: disable=W0201
import sys
import argparse
import time

import yaml
import numpy as np

# torch
import torch
import torch.nn as nn
import torch.optim as optim

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

from .processor import Processor

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    # elif classname.find('Conv2d') != -1:
    elif type(m) == nn.Conv2d:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class REC_Processor(Processor):
    """
        Processor for Skeleton-based Action Recgnition
    """

    def load_model(self):
        self.model = self.io.load_model(self.arg.model,
                                        **(self.arg.model_args))
        self.model.apply(weights_init)
        self.loss = nn.CrossEntropyLoss()  # 交叉熵

    def load_optimizer(self):  # 训练器
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

    def adjust_lr(self):  # 学习率
        if self.arg.optimizer == 'SGD' and self.arg.step:
            lr = self.arg.base_lr * (
                    0.1 ** np.sum(self.meta_info['epoch'] >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr
        else:
            self.lr = self.arg.base_lr

    def show_topk(self, k):
        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
        self.io.print_log('\tTop{}: {:.2f}%'.format(k, 100 * accuracy))
        return '%.2f' % (100 * accuracy)

    def train(self):
        self.model.train()  # 使用BatchNormalizetion()和Dropout()  pytorch中的model.train()
        self.adjust_lr()
        loader = self.data_loader['train']  # 将.npy中的数据传进来
        # print(loader)
        loss_value = []
        i = 0
        accumulation_steps = 4
        start_time = time.time()
        for data, label in loader:
            # get data
            data = data.float().to(self.dev, non_blocking=True)  # tensor  data.shape = (64, 3, 150, 18, 2)在这里加重心点
            # data = data.float().to(self.dev)
            # 将pin_memory(processor.py line 64)开启后，在通过dataloader读取数据后将数据to进GPU时把non_blocking设置为True，可以大幅度加快数据计算的速度。

            # gravity = torch.sum(data, dim=3)  # gravity为第19个关节点,即重心点,取前十八个的平均值
            # gravity /= 18
            # gravity.unsqueeze_(3)  # 补充第三维使得可以和data拼接
            # data = torch.cat((data, gravity), 3) # data.shape = (64, 3, 150, 19, 2)

            label = label.long().to(self.dev)

            # forward
            # output = self.model(data, num_class=self.arg.model_args['num_class'])  # 前向传播
            output = self.model(data)  # 前向传播
            loss = self.loss(output, label)  # 计算损失

            # backward

            # loss.backward()  # 计算梯度
            # # 梯度累计减少显存占用
            # if ((i + 1) % accumulation_steps) == 0:
            #     self.optimizer.step()  # 反向传播， 更新网络参数
            #     self.optimizer.zero_grad()  # 清空梯度
            i += 1
            self.optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 计算梯度
            self.optimizer.step()  # 反向传播， 更新网络参数

            # statistics  #　统计数字
            self.iter_info['loss'] = loss.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            loss_value.append(self.iter_info['loss'])
            self.show_iter_info()
            self.meta_info['iter'] += 1

        self.epoch_info['mean_loss'] = np.mean(loss_value)
        train_loss_one = float(np.mean(loss_value))
        self.show_epoch_info()
        self.io.print_log('Time consumption: ' + str(time.time() - start_time))
        # self.io.print_timer()
        return train_loss_one

    def test(self, evaluation=True):
        self.model.eval()  # 不使用BatchNormalizetion()和Dropout()
        loader = self.data_loader['test']
        loss_value = []
        result_frag = []
        label_frag = []

        for data, label in loader:

            # get data
            # data = data.float().to(self.dev)
            data = data.float().to(self.dev, non_blocking=True)  # tensor  data.shape = (64, 3, 150, 18, 2)在这里加重心点

            # gravity = torch.sum(data, dim=3)  # gravity为第19个关节点,即重心点,取前十八个的平均值
            # gravity /= 18
            # gravity.unsqueeze_(3)  # 补充第三维使得可以和data拼接
            # data = torch.cat((data, gravity), 3)  # data.shape = (64, 3, 150, 19, 2)

            label = label.long().to(self.dev)

            # inference
            with torch.no_grad():
                output = self.model(data)
            result_frag.append(output.data.cpu().numpy())

            # get loss
            if evaluation:
                loss = self.loss(output, label)
                loss_value.append(loss.item())
                label_frag.append(label.data.cpu().numpy())

        self.result = np.concatenate(result_frag)
        if evaluation:
            self.label = np.concatenate(label_frag)
            self.epoch_info['mean_loss'] = np.mean(loss_value)
            val_loss_one = float(np.mean(loss_value))
            self.show_epoch_info()

            # show top-k accuracy
            top1 = None
            for k in self.arg.show_topk:
                if k == 1:
                    top1 = float(self.show_topk(k))
                else:
                    self.show_topk(k)
            return val_loss_one, top1

    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Spatial Temporal Graph Convolution Network')

        # region arguments yapf: disable
        # evaluation
        parser.add_argument('--show_topk', type=int, default=[1, 2], nargs='+',
                            help='which Top K accuracy will be shown')
        # optim
        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+',
                            help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        # endregion yapf: enable

        return parser
