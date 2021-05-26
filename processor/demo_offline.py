#!/usr/bin/env python
import os
import sys
import argparse
import json
import shutil
import time
from sys import platform

import numpy as np
import torch
import skvideo.io

from .my_io import IO
import tools
import tools.utils as utils

import cv2

class DemoOffline(IO):

    def start(self):
        
        # initiate
        label_name_path = './resource/kinetics_skeleton/label_name.txt'
        with open(label_name_path) as f:
            label_name = f.readlines()  # 依次读取每行，label_name为分类类别名称
            label_name = [line.rstrip() for line in label_name]  # 去掉每行头尾空白
            self.label_name = label_name

        # pose estimation
        video, data_numpy = self.pose_estimation()  # data_numpy为视频中识别的骨架数据

        # action recognition
        data = torch.from_numpy(data_numpy)  # 把数组转换成张量，且二者共享内存
        data = data.unsqueeze(0)  # 在第一维上增加一维以符合输入维度
        data = data.float().to(self.dev).detach()  # (1, channel, frame, joint, person)

        # model predict 模型预测、打标签（重要）
        voting_label_name, video_label_name, output, intensity = self.predict(data)

        # render the video 呈现视频
        images = self.render_video(data_numpy, voting_label_name,
                            video_label_name, intensity, video)
        # saving video
        video_name = self.arg.video.split('\\')[-1].split('.')[0]
        output_result_path = '{}/{}.mp4'.format(self.arg.output_dir, video_name)

        writer = skvideo.io.FFmpegWriter(output_result_path, outputdict={'-b': '300000000'})
        # for img in images:
        #     writer.writeFrame(img)
        # writer.close()
        # print('The Demo result has been saved in {}.'.format('D:/PycharmProject/st-gcn/data/demo_result/kick.mp4'))

        # visualize
        for image in images:
            image = image.astype(np.uint8)
            cv2.imshow("ST-GCN", image)
            writer.writeFrame(image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        writer.close()
        print('The Demo result has been saved in {}.'.format(output_result_path))



    def predict(self, data):
        # forward
        output, feature = self.model.extract_feature(data)
        output = output[0]
        feature = feature[0]
        intensity = (feature*feature).sum(dim=0)**0.5  # sum(dim=0)降维，对dim维进行压缩，纵向压缩，纵向值相加，就去掉第dim维
        intensity = intensity.cpu().detach().numpy()  # 将张量转换为numpy格式

        # get result
        # classification result of the full sequence
        voting_label = output.sum(dim=3).sum(
            dim=2).sum(dim=1).argmax(dim=0)  # argmax(dim=0)就是得到dim维最大值的序号索引
        voting_label_name = self.label_name[voting_label]
        # classification result for each person of the latest frame
        num_person = data.size(4)
        latest_frame_label = [output[:, :, :, m].sum(
            dim=2)[:, -1].argmax(dim=0) for m in range(num_person)]  # output[:, :, :, m]取四维数组第四维第m列的所有数据，[:, -1]第二维倒数第一列所有元素（-2为倒数第二列，依次类推）
        latest_frame_label_name = [self.label_name[l]
                                   for l in latest_frame_label]

        num_person = output.size(3)
        num_frame = output.size(1)
        video_label_name = list()
        for t in range(num_frame):
            frame_label_name = list()
            for m in range(num_person):
                person_label = output[:, t, :, m].sum(dim=1).argmax(dim=0)
                person_label_name = self.label_name[person_label]
                frame_label_name.append(person_label_name)
            video_label_name.append(frame_label_name)
        return voting_label_name, video_label_name, output, intensity

    def render_video(self, data_numpy, voting_label_name, video_label_name, intensity, video):
        images = utils.visualization.stgcn_visualize(  # 可视化为四分屏
            data_numpy,
            self.model.graph.edge,
            intensity, video,
            voting_label_name,
            video_label_name,
            self.arg.height)
        return images

    def pose_estimation(self):  # 调用openpose识别出关节点
        # load openpose python api
        # if self.arg.openpose is not None:
        #     sys.path.append('{}/python'.format(self.arg.openpose))
        #     sys.path.append('{}/build/python'.format(self.arg.openpose))
        #     sys.path.append('{}/build/python/openpose'.format(self.arg.openpose))
        #     sys.path.append('{}/build/example/openpose'.format(self.arg.openpose))
        #     sys.path.append('{}/build/python/openpose/Release'.format(self.arg.openpose))
        #     sys.path.append('{}/build/x64/Release'.format(self.arg.openpose))
        #     sys.path.append('{}/build/bin'.format(self.arg.openpose))
        # try:
        #     #from openpose import pyopenpose as op
        #     import pyopenpose as op
        # except:
        #     print('Can not find Openpose Python API.')
        #     return
        #     dir_path = os.path.dirname(os.path.realpath(__file__))

        dir_path = os.path.dirname(os.path.realpath(__file__))
        try:
            # Windows Import
            # if platform == "win32":
            # Change these variables to point to the correct folder (Release/x64 etc.)
            # sys.path.append('D:/PycharmProject/openpose/build/python/openpose/Release')
            # os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + 'D:/PycharmProject/openpose/build/x64/Release' + dir_path + 'D:/PycharmProject/openpose/build/bin'
            sys.path.append(dir_path + '/../../openpose/build/python/openpose/Release');
            os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + '/../../openpose/build/x64/Release;' + dir_path + '/../../openpose/build/bin;'
            import pyopenpose as op
        except ImportError as e:
            print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
            raise e



        video_name = self.arg.video.split('/')[-1].split('.')[0]

        # initiate
        opWrapper = op.WrapperPython()
        params = dict(model_folder='./models', model_pose='COCO')
        opWrapper.configure(params)
        opWrapper.start()
        self.model.eval()
        video_capture = cv2.VideoCapture(self.arg.video)
        # video_capture = cv2.VideoCapture(0) ＃　０为打开摄像头
        video_length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        pose_tracker = naive_pose_tracker(data_frame=video_length)

        # pose estimation
        start_time = time.time()
        frame_index = 0
        video = list()
        while(True):

            # get image
            ret, orig_image = video_capture.read()
            if orig_image is None:
                break
            source_H, source_W, _ = orig_image.shape
            orig_image = cv2.resize(
                orig_image, (256 * source_W // source_H, 256))
            H, W, _ = orig_image.shape
            video.append(orig_image)

            # pose estimation
            datum = op.Datum()
            datum.cvInputData = orig_image
            # opWrapper.emplaceAndPop([datum])
            opWrapper.emplaceAndPop(op.VectorDatum([datum]))
            multi_pose = datum.poseKeypoints  # (num_person, num_joint, 3)

            #if len(multi_pose.shape) != 3:  # 是否三维数组
            if (multi_pose is None):  # 是否有数据，否则为None的时候会报错
                continue

            # normalization
            multi_pose[:, :, 0] = multi_pose[:, :, 0]/W
            multi_pose[:, :, 1] = multi_pose[:, :, 1]/H
            multi_pose[:, :, 0:2] = multi_pose[:, :, 0:2] - 0.5
            multi_pose[:, :, 0][multi_pose[:, :, 2] == 0] = 0
            multi_pose[:, :, 1][multi_pose[:, :, 2] == 0] = 0

            # gravity_pose = np.sum(multi_pose, axis=1)  # 在识别到的骨架点中加入重心点
            # gravity_pose /= 18
            # gravity_pose = np.expand_dims(gravity_pose, 1)
            # multi_pose = np.concatenate((multi_pose, gravity_pose), 1)  # 拼接

            # pose tracking
            pose_tracker.update(multi_pose, frame_index)
            frame_index += 1

            print('Pose estimation ({}/{}).'.format(frame_index, video_length))

        data_numpy = pose_tracker.get_skeleton_sequence()  # 19个骨架序列 data_numpy.shape = (3, 300, 19, 3)
        return video, data_numpy

    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = IO.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Demo for Spatial Temporal Graph Convolution Network')

        # region arguments yapf: disable
        parser.add_argument('--video',
                            default='./resource/media/skateboarding.mp4',
                            help='Path to video')
        parser.add_argument('--openpose',
                            default=None,
                            help='Path to openpose')
        parser.add_argument('--model_input_frame',
                            default=128,
                            type=int)
        parser.add_argument('--model_fps',
                            default=30,
                            type=int)
        parser.add_argument('--height',
                            default=1080,
                            type=int,
                            help='height of frame in the output video.')
        parser.set_defaults(
            config='./config/st_gcn/kinetics-skeleton/demo_offline.yaml')
        parser.set_defaults(print_log=False)
        # endregion yapf: enable

        parser.add_argument('--output_dir',
                            default='./data/demo_result',
                            help='Path to save results')
        return parser

class naive_pose_tracker():
    """ A simple tracker for recording person poses and generating skeleton sequences.
    For actual occasion, I recommend you to implement a robuster tracker.
    Pull-requests are welcomed.
    一个简单的跟踪器，用于记录人的姿势和生成骨骼序列。
    """

    def __init__(self, data_frame=128, num_joint=18, max_frame_dis=np.inf):
        self.data_frame = data_frame
        self.num_joint = num_joint
        self.max_frame_dis = max_frame_dis
        self.latest_frame = 0
        self.trace_info = list()

    def update(self, multi_pose, current_frame):
        # multi_pose.shape: (num_person, num_joint, 3)

        if current_frame <= self.latest_frame:
            return

        if len(multi_pose.shape) != 3:
            return

        score_order = (-multi_pose[:, :, 2].sum(axis=1)).argsort(axis=0)
        for p in multi_pose[score_order]:

            # match existing traces
            matching_trace = None
            matching_dis = None
            for trace_index, (trace, latest_frame) in enumerate(self.trace_info):  # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
                # trace.shape: (num_frame, num_joint, 3)
                if current_frame <= latest_frame:
                    continue
                mean_dis, is_close = self.get_dis(trace, p)
                if is_close:
                    if matching_trace is None:
                        matching_trace = trace_index
                        matching_dis = mean_dis
                    elif matching_dis > mean_dis:
                        matching_trace = trace_index
                        matching_dis = mean_dis

            # update trace information 更新跟踪信息
            if matching_trace is not None:
                trace, latest_frame = self.trace_info[matching_trace]

                # padding zero if the trace is fractured  如果跟踪断开则填充零
                pad_mode = 'interp' if latest_frame == self.latest_frame else 'zero'
                pad = current_frame-latest_frame-1
                new_trace = self.cat_pose(trace, p, pad, pad_mode)
                self.trace_info[matching_trace] = (new_trace, current_frame)

            else:
                new_trace = np.array([p])
                self.trace_info.append((new_trace, current_frame))

        self.latest_frame = current_frame

    def get_skeleton_sequence(self):

        # remove old traces
        valid_trace_index = []
        for trace_index, (trace, latest_frame) in enumerate(self.trace_info):
            if self.latest_frame - latest_frame < self.data_frame:
                valid_trace_index.append(trace_index)
        self.trace_info = [self.trace_info[v] for v in valid_trace_index]

        num_trace = len(self.trace_info)
        if num_trace == 0:
            return None

        data = np.zeros((3, self.data_frame, self.num_joint, num_trace))
        for trace_index, (trace, latest_frame) in enumerate(self.trace_info):
            end = self.data_frame - (self.latest_frame - latest_frame)
            d = trace[-end:]
            beg = end - len(d)
            data[:, beg:end, :, trace_index] = d.transpose((2, 0, 1))

        return data

    # concatenate pose to a trace 将姿势连接成轨迹
    def cat_pose(self, trace, pose, pad, pad_mode):
        # trace.shape: (num_frame, num_joint, 3)
        num_joint = pose.shape[0]
        num_channel = pose.shape[1]
        if pad != 0:
            if pad_mode == 'zero':
                trace = np.concatenate(
                    (trace, np.zeros((pad, num_joint, 3))), 0)
            elif pad_mode == 'interp':
                last_pose = trace[-1]
                coeff = [(p+1)/(pad+1) for p in range(pad)]
                interp_pose = [(1-c)*last_pose + c*pose for c in coeff]
                trace = np.concatenate((trace, interp_pose), 0)
        new_trace = np.concatenate((trace, [pose]), 0)
        return new_trace

    # calculate the distance between a existing trace and the input pose 计算已存在轨迹和输入姿态之间的距离

    def get_dis(self, trace, pose):
        last_pose_xy = trace[-1, :, 0:2]
        curr_pose_xy = pose[:, 0:2]

        mean_dis = ((((last_pose_xy - curr_pose_xy)**2).sum(1))**0.5).mean()
        wh = last_pose_xy.max(0) - last_pose_xy.min(0)
        scale = (wh[0] * wh[1]) ** 0.5 + 0.0001
        is_close = mean_dis < scale * self.max_frame_dis
        return mean_dis, is_close

# 公式改动：加减限定条件，之后验证收敛。界、收敛、速度、拟合速度