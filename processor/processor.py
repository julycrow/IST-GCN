#!/usr/bin/env python
# pylint: disable=W0201
import os
import sys
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import pandas as pd
# torch
import torch

import torch.nn as nn
import torch.optim as optim

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

from .my_io import IO


class Processor(IO):
    """
        Base Processor
    """

    def __init__(self, argv=None):

        self.val_epoch = []
        self.train_epoch = []
        self.top1 = []
        self.val_loss = []
        self.train_loss = []
        self.load_arg(argv)
        self.init_environment()
        self.load_model()
        self.load_weights()
        self.gpu()
        self.load_data()
        self.load_optimizer()

    def init_environment(self):

        super().init_environment()
        self.result = dict()  # 空字典
        self.iter_info = dict()
        self.epoch_info = dict()
        self.meta_info = dict(epoch=0, iter=0)

    def load_optimizer(self):
        pass

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        if 'debug' not in self.arg.train_feeder_args:
            self.arg.train_feeder_args['debug'] = self.arg.debug
        self.data_loader = dict()
        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(  # data_loader['train']里包含npy和pkl两个代表的data和label
                dataset=Feeder(**self.arg.train_feeder_args),  # train.yaml里面
                batch_size=self.arg.batch_size,
                shuffle=True,  # 将输入数据的顺序打乱，是为了使数据更有独立性
                num_workers=self.arg.num_worker * torchlight.ngpu(  # 工作者数量，默认是0。使用多少个子进程来导入数据。
                    self.arg.device),
                # pin_memory=True,
                # 将pin_memory开启后，在通过dataloader(recognition.py line 92)读取数据后将数据to进GPU时把non_blocking设置为True，可以大幅度加快数据计算的速度。
                drop_last=True)  # 丢弃最后数据，默认为False。设置了 batch_size 的数目后，最后一批数据未必是设置的数目，有可能会小些。这时你是否需要丢弃这批数据。
        if self.arg.test_feeder_args:
            self.data_loader['test'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.test_feeder_args),
                batch_size=self.arg.test_batch_size,
                shuffle=False,
                pin_memory=True,
                # 将pin_memory开启后，在通过dataloader(recognition.py line 92)读取数据后将数据to进GPU时把non_blocking设置为True，可以大幅度加快数据计算的速度。
                num_workers=self.arg.num_worker * torchlight.ngpu(
                    self.arg.device))

    def show_epoch_info(self):
        for k, v in self.epoch_info.items():
            self.io.print_log('\t{}: {}'.format(k, v))  # mean_loss: 0.92XXXXXXXXXX
        if self.arg.pavi_log:
            self.io.log('train', self.meta_info['iter'], self.epoch_info)

    def show_iter_info(self):
        if self.meta_info['iter'] % self.arg.log_interval == 0:
            info = '\tIter {} Done.'.format(self.meta_info['iter'])  # Iter 2000 Done. | loss: 0.7397 | lr: 0.001000
            for k, v in self.iter_info.items():
                if isinstance(v, float):
                    info = info + ' | {}: {:.4f}'.format(k, v)
                else:
                    info = info + ' | {}: {}'.format(k, v)

            self.io.print_log(info)

            if self.arg.pavi_log:
                self.io.log('train', self.meta_info['iter'], self.iter_info)

    def train(self):
        for _ in range(100):
            self.iter_info['loss'] = 0
            self.show_iter_info()
            self.meta_info['iter'] += 1
        self.epoch_info['mean loss'] = 0
        self.show_epoch_info()

    def test(self):
        for _ in range(100):
            self.iter_info['loss'] = 1
            self.show_iter_info()
        self.epoch_info['mean loss'] = 1
        self.show_epoch_info()

    def plot_line(self, train_epoch, train_loss, val_epoch, val_loss, top1):
        plt.figure(1)
        plt.plot(train_epoch, train_loss, 'g-', label="train_loss")
        plt.plot(val_epoch, val_loss, 'r-', label="val_loss")
        plt.title('train_loss and val_loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(loc='best')
        plt.savefig(self.arg.work_dir + r'/loss.jpg')
        plt.show()

        plt.figure(2)
        plt.plot(val_epoch, top1, 'b-', label="acc")
        acc_max = top1.index(max(top1))
        show_max = '[' + str(val_epoch[acc_max]) + ' ' + str(top1[acc_max]) + ']'
        plt.plot(val_epoch[acc_max], top1[acc_max], 'go')
        plt.annotate(show_max, xy=(val_epoch[acc_max], top1[acc_max]), xytext=(val_epoch[acc_max], top1[acc_max]))
        # plt.ylim((0, None))  # 纵坐标从0开始
        plt.ylim((0, 100))  # 纵坐标范围:0-100
        plt.title('acc')
        plt.xlabel('epoch')
        plt.ylabel('acc')
        plt.legend(loc='best')
        plt.savefig(self.arg.work_dir + r'/acc.jpg')
        plt.show()
        return top1[acc_max]

    def save_csv(self, train_epoch, train_loss, val_epoch, val_loss, top1):
        max_len = max(len(train_epoch), len(train_loss), len(val_epoch), len(val_loss), len(top1))
        train_epoch.extend((max_len - len(train_epoch)) * [''])
        train_loss.extend((max_len - len(train_loss)) * [''])
        val_epoch.extend((max_len - len(val_epoch)) * [''])
        val_loss.extend((max_len - len(val_loss)) * [''])
        top1.extend((max_len - len(top1)) * [''])
        dataframe = pd.DataFrame(
            {'train_epoch': train_epoch, 'train_loss': train_loss, 'val_epoch': val_epoch, 'val_loss': val_loss,
             'acc': top1})
        dataframe.to_csv(self.arg.work_dir + r"/loss-acc.csv", sep=',')

    def start(self):
        self.io.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
        # training phase
        if self.arg.phase == 'train':
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                self.meta_info['epoch'] = epoch

                # training
                self.io.print_log('Training epoch: {}'.format(epoch))
                # self.train()
                self.train_loss.append(self.train())
                self.train_epoch.append(epoch)

                self.io.print_log('Done.')

                # save model
                if ((epoch + 1) % self.arg.save_interval == 0) or (
                        epoch + 1 == self.arg.num_epoch):
                    filename = 'epoch{}_model.pt'.format(epoch + 1)
                    self.io.save_model(self.model, filename)

                # evaluation
                if ((epoch + 1) % self.arg.eval_interval == 0) or (
                        epoch + 1 == self.arg.num_epoch):
                    self.io.print_log('Eval epoch: {}'.format(epoch))
                    # self.test()
                    val, top = self.test()
                    self.val_loss.append(val)
                    self.top1.append(top)
                    self.val_epoch.append(epoch)
                    self.io.print_log('Done.')

            # 画图
            acc_max = self.plot_line(self.train_epoch, self.train_loss, self.val_epoch, self.val_loss, self.top1)
            # 保存为csv
            self.save_csv(self.train_epoch, self.train_loss, self.val_epoch, self.val_loss, self.top1)
            # 保存路径重命名，包含batch_size, epoch, max_acc的信息
            new_path = self.arg.work_dir + '_' + str(self.arg.batch_size) + '_' + str(self.arg.num_epoch) + '_' + str(
                acc_max)
            os.rename(self.arg.work_dir, new_path)
        # test phase
        elif self.arg.phase == 'test':

            # the path of weights must be appointed
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.io.print_log('Model:   {}.'.format(self.arg.model))
            self.io.print_log('Weights: {}.'.format(self.arg.weights))

            # evaluation
            self.io.print_log('Evaluation Start:')
            self.test()
            self.io.print_log('Done.\n')

            # save the output of model
            if self.arg.save_result:
                result_dict = dict(
                    zip(self.data_loader['test'].dataset.sample_name,
                        self.result))
                self.io.save_pkl(result_dict, 'test_result.pkl')

    @staticmethod
    def get_parser(add_help=False):

        # region arguments yapf: disable
        # parameter priority: command line > config > default
        parser = argparse.ArgumentParser(add_help=add_help, description='Base Processor')

        parser.add_argument('-w', '--work_dir', default='./work_dir/tmp', help='the work folder for storing results')
        parser.add_argument('-c', '--config', default=None, help='path to the configuration file')

        # processor
        parser.add_argument('--phase', default='train', help='must be train or test')
        parser.add_argument('--save_result', type=str2bool, default=False,
                            help='if ture, the output of the model will be stored')
        parser.add_argument('--start_epoch', type=int, default=0, help='start training from which epoch')
        parser.add_argument('--num_epoch', type=int, default=80, help='stop training in which epoch')
        parser.add_argument('--use_gpu', type=str2bool, default=True, help='use GPUs or not')
        parser.add_argument('--device', type=int, default=0, nargs='+',
                            help='the indexes of GPUs for training or testing')

        # visulize and debug
        parser.add_argument('--log_interval', type=int, default=100,
                            help='the interval for printing messages (#iteration)')
        parser.add_argument('--save_interval', type=int, default=10,
                            help='the interval for storing models (#iteration)')
        parser.add_argument('--eval_interval', type=int, default=5,
                            help='the interval for evaluating models (#iteration)')
        parser.add_argument('--save_log', type=str2bool, default=True, help='save logging or not')
        parser.add_argument('--print_log', type=str2bool, default=True, help='print logging or not')
        parser.add_argument('--pavi_log', type=str2bool, default=False, help='logging on pavi or not')

        # feeder
        parser.add_argument('--feeder', default='feeder.feeder', help='data loader will be used')
        parser.add_argument('--num_worker', type=int, default=0, help='the number of worker per gpu for data loader')
        parser.add_argument('--train_feeder_args', action=DictAction, default=dict(),
                            help='the arguments of data loader for training')
        parser.add_argument('--test_feeder_args', action=DictAction, default=dict(),
                            help='the arguments of data loader for test')
        parser.add_argument('--batch_size', type=int, default=256, help='training batch size')
        parser.add_argument('--test_batch_size', type=int, default=256, help='test batch size')
        parser.add_argument('--debug', action="store_true", help='less data, faster loading')

        # model
        parser.add_argument('--model', default=None, help='the model will be used')
        parser.add_argument('--model_args', action=DictAction, default=dict(), help='the arguments of model')
        parser.add_argument('--weights', default=None, help='the weights for network initialization')
        parser.add_argument('--ignore_weights', type=str, default=[], nargs='+',
                            help='the name of weights which will be ignored in the initialization')
        # endregion yapf: enable

        return parser
