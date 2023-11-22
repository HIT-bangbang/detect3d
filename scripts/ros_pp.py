#! /home/bang/anaconda3/envs/openmmlab/bin/python

import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from pyquaternion import Quaternion

from mymsgs.msg import PointWithBBox

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

        self.subLidar = rospy.Subscriber("/points_raw", PointCloud2, self.lidar_callback, queue_size=1, buff_size=2 ** 12)
        self.pubbbox = rospy.Publisher("/detect3d/bbox", BoundingBoxArray, queue_size=1)
        self.pubbboxwithbbox = rospy.Publisher("/detect3d/PointWithbbox", PointWithBBox, queue_size=1)

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

        self.PubPointWithBBox(result,PointCloudsmsg)
        
        callback_end = time.time()
        # 看一下总共的推理时间
        print('callback function used total_time:', (callback_start - callback_end))

    def PubPointWithBBox(self, result, PointCloudsmsg):
        scores = result.pred_instances_3d.scores_3d.cpu().numpy()
        mask = scores > 0.5
        scores = scores[mask]
        boxes_lidar = result.pred_instances_3d.bboxes_3d[mask].cpu().numpy()
        
        if boxes_lidar.shape[0] != 0:
            print(f"Detected {boxes_lidar.shape[0]} boundboxs ")
        label = result.pred_instances_3d.labels_3d[mask].cpu().numpy()


        # 创建一个BoundingBoxArray
        bboxarr = BoundingBoxArray()
        # BoundingBoxArray的参考坐标系就是激光雷达坐标系
        bboxarr.header.frame_id = PointCloudsmsg.header.frame_id
        bboxarr.header.stamp = PointCloudsmsg.header.stamp

        # 将result中所有的BoundingBox塞到BoundingBoxArray里面
        for i in range(boxes_lidar.shape[0]):
            # 创建一个BoundingBox
            bbox = BoundingBox()
            bbox.header.frame_id = PointCloudsmsg.header.frame_id
            bbox.header.stamp = PointCloudsmsg.header.stamp
            bbox.pose.position.x = float(boxes_lidar[i][0])
            bbox.pose.position.y = float(boxes_lidar[i][1])
            bbox.pose.position.z = float(boxes_lidar[i][2])+float(boxes_lidar[i][5])/2
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
            bboxarr.boxes.append(bbox)
        # 发布检测框
        self.pubbbox.publish(bboxarr)
        # 发布点云和检测框二合一消息
        pointswithbbbox = PointWithBBox()
        pointswithbbbox.CloudMsg = PointCloudsmsg
        pointswithbbbox.BBboxArray = bboxarr
        self.pubbboxwithbbox.publish(pointswithbbbox)




if __name__ == '__main__':

    PATH = '/home/bang/mmdetection3d/'
    config_path = PATH + 'configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py'
    checkpoints = PATH + 'checkpoints/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth'
    device = 'cuda:0'
    
    rospy.init_node('mmdetection3d_ros_node', anonymous=True)

    pp = ros_cls(config_path, checkpoints, device)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
