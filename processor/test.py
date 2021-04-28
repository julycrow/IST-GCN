import os
import sys
import argparse
import json
import shutil
import time
from sys import platform

import numpy as np


import cv2


def pose_estimation():  # 调用openpose识别出关节点
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
        os.environ['PATH'] = os.environ[
                                 'PATH'] + ';' + dir_path + '/../../openpose/build/x64/Release;' + dir_path + '/../../openpose/build/bin;'
        import pyopenpose as op
    except ImportError as e:
        print(
            'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e

    #video_name = self.arg.video.split('/')[-1].split('.')[0]

    # initiate
    opWrapper = op.WrapperPython()
    params = dict(model_folder='../models', model_pose='COCO')
    opWrapper.configure(params)
    opWrapper.start()
    #self.model.eval()
    video_capture = cv2.VideoCapture('C:\\Users\\july\\Desktop\\thrust_in_side_kick_Trim.mp4')
    # video_capture = cv2.VideoCapture(0)
    video_length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    pose_tracker = naive_pose_tracker(data_frame=video_length)

    # pose estimation
    # start_time = time.time()
    frame_index = 0
    video = list()
    while (True):

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

        # if len(multi_pose.shape) != 3:  # 是否三维数组
        if (multi_pose is None):  # 是否有数据，否则为None的时候会报错
            continue

        # normalization
        multi_pose[:, :, 0] = multi_pose[:, :, 0] / W
        multi_pose[:, :, 1] = multi_pose[:, :, 1] / H
        multi_pose[:, :, 0:2] = multi_pose[:, :, 0:2] - 0.5
        multi_pose[:, :, 0][multi_pose[:, :, 2] == 0] = 0
        multi_pose[:, :, 1][multi_pose[:, :, 2] == 0] = 0

        # pose tracking
        pose_tracker.update(multi_pose, frame_index)
        frame_index += 1

        print('Pose estimation ({}/{}).'.format(frame_index, video_length))

    data_numpy = pose_tracker.get_skeleton_sequence()
    return video, data_numpy


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
            for trace_index, (trace, latest_frame) in enumerate(
                    self.trace_info):  # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
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
                pad = current_frame - latest_frame - 1
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
                coeff = [(p + 1) / (pad + 1) for p in range(pad)]
                interp_pose = [(1 - c) * last_pose + c * pose for c in coeff]
                trace = np.concatenate((trace, interp_pose), 0)
        new_trace = np.concatenate((trace, [pose]), 0)
        return new_trace

    # calculate the distance between a existing trace and the input pose 计算已存在轨迹和输入姿态之间的距离

    def get_dis(self, trace, pose):
        last_pose_xy = trace[-1, :, 0:2]
        curr_pose_xy = pose[:, 0:2]

        mean_dis = ((((last_pose_xy - curr_pose_xy) ** 2).sum(1)) ** 0.5).mean()
        wh = last_pose_xy.max(0) - last_pose_xy.min(0)
        scale = (wh[0] * wh[1]) ** 0.5 + 0.0001
        is_close = mean_dis < scale * self.max_frame_dis
        return mean_dis, is_close


video, data_numpy = pose_estimation()
print(data_numpy)
