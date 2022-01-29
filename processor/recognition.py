#!/usr/bin/env python
# pylint: disable=W0201
import sys
import argparse
import time
import os
import json

import yaml
import numpy as np

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import numpy
# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

from .processor import Processor
import matplotlib.pyplot as plt
import itertools
from prettytable import PrettyTable
# from torch.utils.tensorboard import SummaryWriter
import pandas as pd


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


class ConfusionMatrix(object):
    """
    绘制混淆矩阵
    注意，如果显示的图像不全，是matplotlib版本问题
    本例程使用matplotlib-3.2.1(windows and ubuntu)绘制正常
    需要额外安装prettytable库
    """

    def __init__(self, num_classes: int, labels: list, workdir: str, epoch: int):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels
        self.workdir = workdir
        self.epoch = epoch

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):
        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("the model accuracy is ", acc)

        # precision, recall, specificity
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            table.add_row([self.labels[i], Precision, Recall, Specificity])
        print(table)

    def plot(self):
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        # plt.xticks(range(self.num_classes), self.labels, rotation=45)
        # plt.xticks(range(self.num_classes), range(self.num_classes), rotation=45)
        # 设置y轴坐标label
        # plt.yticks(range(self.num_classes), self.labels)
        # plt.yticks(range(self.num_classes), range(self.num_classes))
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix')

        # 在图中标注数量/概率信息
        # thresh = matrix.max() / 2
        # for x in range(self.num_classes):
        #     for y in range(self.num_classes):
        #         # 注意这里的matrix[y, x]不是matrix[x, y]
        #         info = int(matrix[y, x])
        #         if info:
        #             plt.text(x, y, info,
        #                      verticalalignment='center',
        #                      horizontalalignment='center',
        #                      color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.show()
        plt.savefig(self.workdir + '/' + str(self.epoch) + 'confusion_matrix.jpg')
        add_pre = matrix  # 矩阵copy，后面会加入该行精确率
        pre = []  # 精确率

        for i in range(self.num_classes):
            cl_sum = 0
            for j in add_pre[i]:
                cl_sum += j
            if cl_sum == 0:
                i_acc = 0 * 1.0000
            else:
                i_acc = add_pre[i][i] * 1.0000 / cl_sum

            # add_pre[i][self.num_classes] = i_acc
            pre.append(i_acc)

        pre = np.array(pre)
        pre = pre.reshape(self.num_classes, 1)
        add_pre = np.concatenate((add_pre, pre), axis=1)  # 精确率拼接在最后一列
        dataframe = pd.DataFrame(add_pre)  # 二维表
        dataframe.to_csv(self.workdir + r"/matrix" + str(self.epoch) + r".csv", sep=',')


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
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]  # 预测正确为1，错误为0
        accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
        self.io.print_log('\tTop{}: {:.2f}%'.format(k, 100 * accuracy))
        return '%.2f' % (100 * accuracy)

    def train(self):
        self.model.train()  # 使用BatchNormalizetion()和Dropout()  pytorch中的model.train()
        self.adjust_lr()
        loader = self.data_loader['train']  # 将.npy中的数据传进来
        # print(loader)
        loss_value = []

        # accumulation_steps = 4
        # conf_matrix = torch.zeros(5, 5)

        # def confusion_matrix(preds, labels, conf_matrix):
        #     for p, t in zip(preds, labels):
        #         conf_matrix[p, t] += 1
        #     return conf_matrix
        # def plot_confusion_matrix(cm, classes, epoch, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        #     '''
        #     This function prints and plots the confusion matrix.
        #     Normalization can be applied by setting `normalize=True`.
        #     Input
        #     - cm : 计算出的混淆矩阵的值
        #     - classes : 混淆矩阵中每一行每一列对应的列
        #     - normalize : True:显示百分比, False:显示个数
        #     '''
        #
        #     if normalize:
        #         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #         print("Normalized confusion matrix")
        #     else:
        #         print('Confusion matrix, without normalization')
        #     print(cm)
        #     plt.imshow(cm, interpolation='nearest', cmap=cmap)
        #     plt.title(title)
        #     plt.colorbar()
        #     tick_marks = np.arange(len(classes))
        #     plt.xticks(tick_marks, classes, rotation=90)
        #     plt.yticks(tick_marks, classes)
        #
        #     # 。。。。。。。。。。。。新增代码开始处。。。。。。。。。。。。。。。。
        #     # x,y轴长度一致(问题1解决办法）
        #     plt.axis("equal")
        #     # x轴处理一下，如果x轴或者y轴两边有空白的话(问题2解决办法）
        #     ax = plt.gca()  # 获得当前axis
        #     left, right = plt.xlim()  # 获得x轴最大最小值
        #     ax.spines['left'].set_position(('data', left))
        #     ax.spines['right'].set_position(('data', right))
        #     for edge_i in ['top', 'bottom', 'right', 'left']:
        #         ax.spines[edge_i].set_edgecolor("white")
        #     # 。。。。。。。。。。。。新增代码结束处。。。。。。。。。。。。。。。。
        #
        #     thresh = cm.max() / 2.
        #     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        #         num = '{:.2f}'.format(cm[i, j]) if normalize else int(cm[i, j])
        #         plt.text(j, i, num,
        #                  verticalalignment='center',
        #                  horizontalalignment="center",
        #                  color="white" if float(num) > float(thresh) else "black")
        #     plt.tight_layout()
        #     plt.ylabel('True label')
        #     plt.xlabel('Predicted label')
        #     plt.savefig(self.arg.work_dir + r'/matrix_' +epoch + r'.jpg')
        #     plt.show()

        start_time = time.time()

        for data, label in loader:
            # get data

            # 转为numpy方便删除左半边
            # data = data.numpy()  # shape (16, 3, 150, 25, 2)
            # del_half = [8, 9, 10, 11, 16, 17, 18, 19, 23, 24]
            # data = np.delete(data, del_half, axis=3)
            # data = torch.from_numpy(data)  # shape (16, 3, 150, 15, 2)

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

            # prediction = torch.max(output, 1)[1]
            # conf_matrix = confusion_matrix(prediction, labels=label, conf_matrix=conf_matrix)

            loss = self.loss(output, label)  # 计算损失

            # backward

            # loss.backward()  # 计算梯度
            # # 梯度累计减少显存占用
            # if ((i + 1) % accumulation_steps) == 0:
            #     self.optimizer.step()  # 反向传播， 更新网络参数
            #     self.optimizer.zero_grad()  # 清空梯度
            self.optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 计算梯度
            self.optimizer.step()  # 反向传播， 更新网络参数

            # statistics  #　统计数字
            self.iter_info['loss'] = loss.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            loss_value.append(self.iter_info['loss'])
            self.show_iter_info()
            self.meta_info['iter'] += 1

            self.writer.add_scalar('train_loss', self.iter_info['loss'], self.meta_info['iter'])

        self.epoch_info['mean_loss'] = np.mean(loss_value)

        train_loss_one = float(np.mean(loss_value))
        self.show_epoch_info()
        self.io.print_log('Time consumption: ' + str(time.time() - start_time))
        # self.io.print_timer()
        # plot_confusion_matrix(conf_matrix.numpy(), classes=['Pick', 'Sow', 'Spray', 'Hoe', 'Cut'], normalize=False,
        #                       title='Normalized confusion matrix', epoch=self.epoch_info)
        # plot_confusion_matrix(conf_matrix.numpy(), classes=['Pick', 'Sow', 'Spray', 'Hoe', 'Cut'], normalize=True,
        #                       title='Normalized confusion matrix', epoch=self.epoch_info+'_')
        return train_loss_one

    def test(self, evaluation=True):
        self.model.eval()  # 不使用BatchNormalizetion()和Dropout()
        loader = self.data_loader['test']
        loss_value = []
        result_frag = []
        label_frag = []

        # 混淆矩阵
        txt_label_path = './resource/kinetics_skeleton/label_name_ntu60.txt'
        assert os.path.exists(txt_label_path), "cannot find {} file".format(txt_label_path)
        txt_file = open(txt_label_path, 'r')
        labels = txt_file.read().splitlines()
        txt_file.close()
        confusion = ConfusionMatrix(num_classes=int(len(labels)), labels=labels, workdir=self.arg.work_dir,
                                    epoch=self.meta_info['epoch'])

        for data, label in loader:

            # data = data.numpy()  # shape (16, 3, 150, 25, 2)
            # del_half = [8, 9, 10, 11, 16, 17, 18, 19, 23, 24]
            # data = np.delete(data, del_half, axis=3)
            # data = torch.from_numpy(data)  # shape (16, 3, 150, 15, 2)

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

                # self.writer.add_scalar('val_loss', loss, self.meta_info['iter'])

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

            # 求result每行最大概率对应的类别，与label对应
            # rank = self.result.argsort()
            cl_result = [i.argmax() for i in self.result]
            confusion.update(numpy.array(cl_result), self.label)
            confusion.plot()
            # confusion.summary()

            self.writer.add_scalar('val_loss', self.epoch_info['mean_loss'], self.meta_info['epoch'])
            self.writer.add_scalar('acc', top1, self.meta_info['epoch'])

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
        parser.add_argument('--show_topk', type=int, default=[1, 5], nargs='+',
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
