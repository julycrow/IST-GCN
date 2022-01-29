#!/usr/bin/env python
import os
import sys
import argparse
import json
import shutil
import time

import numpy as np
import torch
import skvideo.io

from .my_io import IO
import tools
import tools.utils as utils

import cv2

from queue import Queue
from Neo4j.visualize import visualize
from PIL import Image, ImageDraw, ImageFont


class DemoRealtime(IO):
    """ A demo for utilizing st-gcn in the realtime action recognition.
    The Openpose python-api is required for this demo.

    Since the pre-trained model is trained on videos with 30fps,
    and Openpose is hard to achieve this high speed in the single GPU,
    if you want to predict actions by **camera** in realtime,
    either data interpolation or new pre-trained model
    is required.

    Pull requests are always welcome.
    """

    def start(self):
        # # load openpose python api
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
        dir_path = os.path.dirname(os.path.realpath(__file__))

        warning_person = 2  # 聚集人数
        long_interval = 5400  # 帧间隔(长),3分钟包含的帧数, 用来判断长警情,
        short_interval = 1800  # 帧间隔(短),1分钟包含的帧数, 用来判断短警情
        long_recognize_windows = 180  # 识别窗口(长),用来判断长警情,出现该警情的次数
        short_recognize_windows = 45  # 识别窗口(短),用来判断短警情
        person_counting = smash_counting = pull_counting = fall_counting = 0  # 长时间警情持续帧数
        long_last = 180  # 长时间警情显示持续帧数
        person_list = []  # 维护这个列表来判断是否为长短警情
        smash_list = []
        pull_list = []
        fall_list = []

        try:
            sys.path.append(dir_path + '/../../openpose/build/python/openpose/Release');
            os.environ['PATH'] = os.environ[
                                     'PATH'] + ';' + dir_path + '/../../openpose/build/x64/Release;' + dir_path + '/../../openpose/build/bin;'
            import pyopenpose as op
        except ImportError as e:
            print(
                'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
            raise e

        video_name = self.arg.video.split('/')[-1].split('.')[0]
        label_name_path = './resource/kinetics_skeleton/label_name.txt'
        with open(label_name_path) as f:
            label_name = f.readlines()
            label_name = [line.rstrip() for line in label_name]
            self.label_name = label_name

        # initiate
        opWrapper = op.WrapperPython()
        params = dict(model_folder='./models', model_pose='COCO')
        opWrapper.configure(params)
        opWrapper.start()
        self.model.eval()
        pose_tracker = naive_pose_tracker()

        if self.arg.video == 'camera_source':
            video_capture = cv2.VideoCapture(0)
            # video_capture = cv2.VideoCapture('rtsp://admin:okwy1234@192.168.1.64:554')
            # video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        else:
            video_capture = cv2.VideoCapture(self.arg.video)

        # start recognition
        start_time = time.time()
        frame_index = 0
        # saving video
        video_name = self.arg.video.split('\\')[-1].split('.')[0]
        output_result_path = '{}/{}.mp4'.format(self.arg.output_dir, video_name)

        writer = skvideo.io.FFmpegWriter(output_result_path, outputdict={})
        # writer = skvideo.io.FFmpegWriter(output_result_path)
        while True:

            tic = time.time()

            # get image
            ret, orig_image = video_capture.read()
            if orig_image is None:
                break
            source_H, source_W, _ = orig_image.shape
            orig_image = cv2.resize(
                orig_image, (256 * source_W // source_H, 256))
            H, W, _ = orig_image.shape

            # pose estimation
            datum = op.Datum()
            datum.cvInputData = orig_image
            # opWrapper.emplaceAndPop([datum])
            opWrapper.emplaceAndPop(op.VectorDatum([datum]))
            multi_pose = datum.poseKeypoints  # (num_person, num_joint, 3) 关节点数据
            # if (multi_pose is None) | (len(multi_pose.shape) != 3):  # 是否三维数组
            if multi_pose is None:  # 是否三维数组
                continue

            # 人员聚集识别->识别人数超过规定人数,且持续一段时间
            def return_inf(input, occ_time, orig_image, color_input):

                red = (255, 0, 0)
                green = (0, 255, 0)
                if color_input == 'red':
                    color = red
                else:
                    color = green
                path = visualize(input)  # graph_base.html的path(知识图谱)
                # 输出图像, 时间
                whole_time = time.asctime(time.localtime(occ_time))
                # cv2.destroyAllWindows()
                if isinstance(orig_image, np.ndarray):  # 判断是否OpenCV图片类型
                    orig_image = Image.fromarray(cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB))
                # 创建一个可以在给定图像上绘图的对象
                draw = ImageDraw.Draw(orig_image)
                # 字体的格式
                fontStyle = ImageFont.truetype(
                    "font/simsun.ttc", 20, encoding="utf-8")
                # 绘制文本
                draw.text((10, 10), input, color, fontStyle)
                # 转换回OpenCV格式
                orig_image = cv2.cvtColor(np.asarray(orig_image), cv2.COLOR_RGB2BGR)
                cv2.imshow('', orig_image)
                # cv2.imshow(input.encode("gbk").decode(errors="ignore"), orig_image)
                return path, whole_time, orig_image

            num_person = multi_pose.shape[0]
            if num_person >= warning_person:
                occ_time = time.time()
                if len(person_list) < short_recognize_windows:
                    person_list.append(occ_time)
                elif long_recognize_windows > len(person_list) >= short_recognize_windows:
                    if occ_time - person_list[0] < short_interval:  # 第一次出现与这次出现时间间隔小于1800帧(1分钟),短警情
                        # 添加调用函数输出时间,画面,知识图谱
                        path, whole_time, cur_image = return_inf('短时间非法聚集', occ_time, orig_image, 'green')
                    person_list.append(occ_time)
                else:
                    if occ_time - person_list[0] < long_interval:  # 长警情
                        if person_counting == long_last:
                            person_list.clear()
                            person_counting = 0
                        else:
                            person_counting += 1
                        path, whole_time, cur_image = return_inf('长时间非法聚集', occ_time, orig_image, 'red')
                    elif occ_time - person_list[0] < short_interval:  # 短警情

                        path, whole_time, cur_image = return_inf('短时间非法聚集', occ_time, orig_image, 'green')
                    person_list.append(occ_time)

            # normalization
            multi_pose[:, :, 0] = multi_pose[:, :, 0] / W
            multi_pose[:, :, 1] = multi_pose[:, :, 1] / H
            multi_pose[:, :, 0:2] = multi_pose[:, :, 0:2] - 0.5
            multi_pose[:, :, 0][multi_pose[:, :, 2] == 0] = 0
            multi_pose[:, :, 1][multi_pose[:, :, 2] == 0] = 0

            # pose tracking
            if self.arg.video == 'camera_source':
                frame_index = int((time.time() - start_time) * self.arg.model_fps)
            else:
                frame_index += 1
            pose_tracker.update(multi_pose, frame_index)
            data_numpy = pose_tracker.get_skeleton_sequence()
            data = torch.from_numpy(data_numpy)
            data = data.unsqueeze(0)
            data = data.float().to(self.dev).detach()  # (1, channel, frame, joint, person)

            # model predict
            voting_label_name, video_label_name, output, intensity = self.predict(
                data)

            # 长短警情报警（打砸、推搡、跌倒）
            def longshort_alarm(input, alarm_list, voting_label_name, counting):
                path = whole_time = cur_image = None
                short_input = '短时间' + input
                long_input = '长时间' + input
                occ_time = time.time()
                if len(alarm_list) < short_recognize_windows:
                    alarm_list.append(occ_time)
                elif long_recognize_windows > len(alarm_list) >= short_recognize_windows:
                    if occ_time - alarm_list[0] < short_interval:  # 第一次出现与这次出现时间间隔小于1800帧(1分钟),短警情
                        # 调用return_inf函数输出知识图谱,时间,画面
                        path, whole_time, cur_image = return_inf(short_input, occ_time, orig_image, 'green')

                    alarm_list.append(occ_time)
                else:
                    if occ_time - alarm_list[0] < long_interval:  # 长警情
                        if counting == long_last:
                            alarm_list.clear()
                            counting = 0
                        else:
                            counting += 1
                        path, whole_time, cur_image = return_inf(long_input, occ_time, orig_image, 'red')
                    elif occ_time - alarm_list[0] < short_interval:  # 短警情
                        path, whole_time, cur_image = return_inf(short_input, occ_time, orig_image, 'green')
                    person_list.append(occ_time)
                return path, whole_time, cur_image, counting

            if voting_label_name == 'Pull':
                path, whole_time, cur_image, pull_counting = longshort_alarm('打架', pull_list, voting_label_name, pull_counting)
            if voting_label_name == 'Smash':
                path, whole_time, cur_image, smash_counting = longshort_alarm('打砸', smash_list, voting_label_name, smash_counting)
            if voting_label_name == 'Fall':
                path, whole_time, cur_image, fall_counting = longshort_alarm('摔倒', fall_list, voting_label_name, fall_counting)
            # visualization
            app_fps = 1 / (time.time() - tic)
            image = self.render(data_numpy, voting_label_name,
                                video_label_name, intensity, orig_image, app_fps)
            writer.writeFrame(image)
            cv2.imshow("ST-GCN", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        writer.close()
        print('The Demo result has been saved in {}.'.format(output_result_path))

    def predict(self, data):
        # forward
        output, feature = self.model.extract_feature(data)
        output = output[0]
        feature = feature[0]
        intensity = (feature * feature).sum(dim=0) ** 0.5
        intensity = intensity.cpu().detach().numpy()

        # get result
        # classification result of the full sequence
        voting_label = output.sum(dim=3).sum(
            dim=2).sum(dim=1).argmax(dim=0)
        voting_label_name = self.label_name[voting_label]
        # classification result for each person of the latest frame
        num_person = data.size(4)
        latest_frame_label = [output[:, :, :, m].sum(
            dim=2)[:, -1].argmax(dim=0) for m in range(num_person)]
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

    def render(self, data_numpy, voting_label_name, video_label_name, intensity, orig_image, fps=0):
        images = utils.visualization.stgcn_visualize(
            data_numpy[:, [-1]],
            self.model.graph.edge,
            intensity[[-1]], [orig_image],
            voting_label_name,
            [video_label_name[-1]],
            self.arg.height,
            fps=fps)
        image = next(images)
        image = image.astype(np.uint8)
        return image

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
        parser.add_argument('--police',
                            default=True,
                            type=bool,
                            help='Output alarm information or not.')
        parser.add_argument('--output_dir',
                            default='./data/demo_result',
                            help='Path to save results')
        parser.set_defaults(
            config='./config/st_gcn/kinetics-skeleton/demo_realtime.yaml')
        parser.set_defaults(print_log=False)
        # endregion yapf: enable

        return parser


class naive_pose_tracker():
    """ A simple tracker for recording person poses and generating skeleton sequences.
    For actual occasion, I recommend you to implement a robuster tracker.
    Pull-requests are welcomed.
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
            for trace_index, (trace, latest_frame) in enumerate(self.trace_info):
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

            # update trace information
            if matching_trace is not None:
                trace, latest_frame = self.trace_info[matching_trace]

                # padding zero if the trace is fractured
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

    # concatenate pose to a trace
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

    # calculate the distance between a existing trace and the input pose

    def get_dis(self, trace, pose):
        last_pose_xy = trace[-1, :, 0:2]
        curr_pose_xy = pose[:, 0:2]

        mean_dis = ((((last_pose_xy - curr_pose_xy) ** 2).sum(1)) ** 0.5).mean()
        wh = last_pose_xy.max(0) - last_pose_xy.min(0)
        scale = (wh[0] * wh[1]) ** 0.5 + 0.0001
        is_close = mean_dis < scale * self.max_frame_dis
        return mean_dis, is_close
