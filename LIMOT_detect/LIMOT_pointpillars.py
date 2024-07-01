#! /home/bang/anaconda3/envs/openmmlab/bin/python

import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from pyquaternion import Quaternion

from std_msgs.msg import Float64MultiArray

import time
import numpy as np
# np.float = np.float64
# import ros_numpy
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
        print('[+]-------->Model has been inited!')

        self.subLidar = rospy.Subscriber("/points_raw", PointCloud2, self.lidar_callback,tcp_nodelay=True)
        # self.pubbbox = rospy.Publisher("/detect3d/bbox", BoundingBoxArray, queue_size=1)
        # self.pubbboxwithbbox = rospy.Publisher("/detect3d/PointWithbbox",PointWithBBox, queue_size=10)

        self.pubobs = rospy.Publisher("/detect3d", Float64MultiArray, queue_size=1)

        print('[+]-------->ROS is starting!')

    def lidar_callback(self, PointCloudsmsg):
        callback_start =  time.time()
        points = np.array(list(pc2.read_points(PointCloudsmsg, field_names=("x", "y", "z", "intensity"), skip_nans=True)))
        points = points.astype('float32')
        
        inference_start = time.time()
        
        # 开始推理，结果保存在result中
        torch.cuda.synchronize()
        result, data = inference_detector(self.model, points)
        torch.cuda.synchronize()

        inference_end = time.time()
        
        # 看一下单帧推理时间
        print('single frame use:', (inference_start - inference_end))

        self.PubOBS(result,PointCloudsmsg.header.stamp.to_sec())
        
        callback_end = time.time()
        # 看一下总共的推理时间
        print('callback function used total_time:', (callback_start - callback_end))

    def PubOBS(self, result, timestamp):
        scores = result.pred_instances_3d.scores_3d.cpu().numpy()
        mask = scores > 0.6
        scores = scores[mask]
        boxes_lidar = result.pred_instances_3d.bboxes_3d[mask].cpu().numpy()
        
        if boxes_lidar.shape[0] != 0:
            print(f"Detected {boxes_lidar.shape[0]} boundboxs ")
            label = result.pred_instances_3d.labels_3d[mask].cpu().numpy()

            obs = Float64MultiArray()
            obs.data.append(float(timestamp))
            for i in range(boxes_lidar.shape[0]):
                obs.data.append(float(label[i]))
                obs.data.append(float(boxes_lidar[i][0]))
                obs.data.append(float(boxes_lidar[i][1]))
                obs.data.append(float(boxes_lidar[i][2]))
                obs.data.append(float(boxes_lidar[i][3]))
                obs.data.append(float(boxes_lidar[i][4]))
                obs.data.append(float(boxes_lidar[i][5]))
                obs.data.append(float(boxes_lidar[i][6]))
                obs.data.append(scores[i])

            self.pubobs.publish(obs)
        else:
            pass

#[timestamp, [type, x, y, z, l, w, h, yaw, score], ...,[type ,x, ..., score]].

if __name__ == '__main__':

    PATH = '/home/bang/mmdetection3d/'

    # pointpillars
    config_path = PATH + 'configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py'
    checkpoints = PATH + 'checkpoints/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth'
    
    # # point_rcnn 0.3
    # config_path = PATH + 'configs/point_rcnn/point-rcnn_8xb2_kitti-3d-3class.py'
    # checkpoints = PATH + 'checkpoints/point_rcnn_2x8_kitti-3d-3classes_20211208_151344.pth'

    # # 3dssd 0.065
    # config_path = PATH + 'configs/3dssd/3dssd_4xb4_kitti-3d-car.py'
    # checkpoints = PATH + 'checkpoints/3dssd_4x4_kitti-3d-car_20210818_203828-b89c8fc4.pth'

    # # hv_PartA2 0.12
    # config_path = PATH + 'configs/parta2/parta2_hv_secfpn_8xb2-cyclic-80e_kitti-3d-car.py'
    # checkpoints = PATH + 'checkpoints/hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-car_20210831_022017-cb7ff621.pth'

    # # second 0.065
    # config_path = PATH + 'configs/second/second_hv_secfpn_8xb6-80e_kitti-3d-car.py'
    # checkpoints = PATH + 'checkpoints/hv_second_secfpn_fp16_6x8_80e_kitti-3d-car_20200924_211301-1f5ad833.pth'

    # # pv_rcnn 0.2
    # config_path = PATH + 'configs/pv_rcnn/pv_rcnn_8xb2-80e_kitti-3d-3class.py'
    # checkpoints = PATH + 'checkpoints/pv_rcnn_8xb2-80e_kitti-3d-3class_20221117_234428-b384d22f.pth'


    device = 'cuda:0'
    
    rospy.init_node('mmdetection3d_ros_node', anonymous=True)

    pp = ros_cls(config_path, checkpoints, device)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
