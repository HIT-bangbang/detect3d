#! /home/cstg/anaconda3/envs/solio/bin/python

import rospy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pc2
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from pyquaternion import Quaternion
import glob
import argparse
from pathlib import Path

import time
import numpy as np
import scipy.linalg as linalg
import torch

import sys

sys.path.append("/home/bang/mmdetection3d")

from mmdet3d.apis import (inference_detector, init_model)
from mmdet3d.structures.points import get_points_type


class ros_cls:
    def __init__(self, config_path, checkpoint, device):
        self.config = config_path
        self.checkpoint = checkpoint
        self.device = device
        self.model = init_model(config_path, checkpoint, device)
        self.cb_toc = rospy.Time.now().to_sec()
        self.cb_tic = rospy.Time.now().to_sec()

    def lidar_callback(self, msg):
        print("[+] --------> MSG RECEIVED!")
        data_tic = rospy.Time.now().to_sec()
        points = torch.tensor(pc2.read_points_list(msg, field_names=("x", "y", "z"), skip_nans=True))
        data_toc = rospy.Time.now().to_sec()
        print(f"Load data consumes {(data_toc - data_tic) * 1000} ms")
        # 这个地方占用了200ms的时间
        # 其中torch.tensor转化占用了190ms的时间

        # points_class = get_points_type('LIDAR')
        # points_mmdet3d = points_class(points, points_dim=points.shape[-1], attribute_dims=None)
        torch.cuda.synchronize()
        tic = time.perf_counter()
        result, data = inference_detector(self.model, points)
        torch.cuda.synchronize()
        toc = time.perf_counter()
        print(f'single frame: {(toc - tic) * 1000:.1f} ms')

        scores = result.pred_instances_3d.scores_3d.cpu().numpy()
        mask = scores > 0.05
        scores = scores[mask]
        boxes_lidar = result.pred_instances_3d.bboxes_3d[mask].cpu().numpy()
        if boxes_lidar.shape[0] != 0:
            print(f"boxes size is: {boxes_lidar.shape[0]} ")
        label = result.pred_instances_3d.labels_3d[mask].cpu().numpy()
        arr_bbox = BoundingBoxArray()

        for i in range(boxes_lidar.shape[0]):
            bbox = BoundingBox()
            bbox.header.frame_id = msg.header.frame_id
            bbox.header.stamp = msg.header.stamp
            bbox.pose.position.x = float(boxes_lidar[i][0])
            bbox.pose.position.y = float(boxes_lidar[i][1])
            bbox.pose.position.z = 0.5 + float(boxes_lidar[i][2])
            bbox.dimensions.x = float(boxes_lidar[i][3])
            bbox.dimensions.y = float(boxes_lidar[i][4])
            bbox.dimensions.z = float(boxes_lidar[i][5])
            q = Quaternion(axis=(0, 0, 1), radians=float(boxes_lidar[i][6]))
            bbox.pose.orientation.x = q.x
            bbox.pose.orientation.y = q.y
            bbox.pose.orientation.z = q.z
            bbox.pose.orientation.w = q.w
            bbox.value = scores[i]
            bbox.label = label[i]
            arr_bbox.boxes.append(bbox)
        arr_bbox.header.frame_id = msg.header.frame_id
        pubbbox.publish(arr_bbox)

        # if len(arr_bbox.boxes) != 0:

        pubPoint.publish(msg)

        self.cb_toc = rospy.Time.now().to_sec()
        print(f"Every Frame consume {(self.cb_toc - self.cb_tic) * 1000} ms")
        self.cb_tic = self.cb_toc


if __name__ == '__main__':
    PATH = '/home/bang/mmdetection3d/'
    config_path = PATH + 'configs/votenet/votenet_8xb16_sunrgbd-3d.py'
    checkpoints = PATH + 'checkpoints/votenet_16x8_sunrgbd-3d-10class_20210820_162823-bf11f014.pth'
    device = 'cuda:0'

    global votenet

    rospy.init_node('mmdetection3d_ros_node', anonymous=True)

    votenet = ros_cls(config_path, checkpoints, device)

    print('[+]-------->Loading Model is starting!')

    subLidar = rospy.Subscriber("/velodyne_points", PointCloud2, votenet.lidar_callback, queue_size=1, buff_size=2 ** 12)
    pubbbox = rospy.Publisher("/det3d/bbox", BoundingBoxArray, queue_size=10)
    pubPoint = rospy.Publisher("/lidar_points", PointCloud2, queue_size=1)
    print('[+]-------->ROS is starting!')

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
