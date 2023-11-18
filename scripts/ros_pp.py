#! /home/cstg/anaconda3/envs/solio/bin/python

import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from pyquaternion import Quaternion

 
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

    def lidar_callback(self, msg):
        total_time_start =  time.time()
        points = np.array(list(pc2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True)))
        points = points.astype('float32')
        # points_class = get_points_type('LIDAR')
        # points_mmdet3d = points_class(points, points_dim=points.shape[-1], attribute_dims=None)
        torch.cuda.synchronize()
        tic = time.time()
        result, data = inference_detector(self.model, points)
        torch.cuda.synchronize()
        toc = time.time()
        print('single frame:', (toc - tic))

        scores = result.pred_instances_3d.scores_3d.cpu().numpy()
        mask = scores > 0.5
        scores = scores[mask]
        boxes_lidar = result.pred_instances_3d.bboxes_3d[mask].cpu().numpy()
        if boxes_lidar.shape[0] != 0:
            print(f"boxes size is: {boxes_lidar.shape[0]} ")
        label = result.pred_instances_3d.labels_3d[mask].cpu().numpy()
        arr_bbox = BoundingBoxArray()
        arr_bbox.header.frame_id = msg.header.frame_id

        for i in range(boxes_lidar.shape[0]):
            bbox = BoundingBox()
            bbox.header.frame_id = msg.header.frame_id
            bbox.header.stamp = msg.header.stamp
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
            arr_bbox.boxes.append(bbox)
        pubbbox.publish(arr_bbox)
        
        total_time_end = time.time()
        print('total_time:', (total_time_start - total_time_end))

        # if len(arr_bbox.boxes) != 0:

        # pubPoint.publish(msg)




if __name__ == '__main__':

    PATH = '/home/bang/mmdetection3d/'
    config_path = PATH + 'configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py'
    checkpoints = PATH + 'checkpoints/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth'
    device = 'cuda:0'

    global pp

    pp = ros_cls(config_path, checkpoints, device)
    print('[+]-------->Loading Model is starting!')

    rospy.init_node('mmdetection3d_ros_node', anonymous=True)

    subLidar = rospy.Subscriber("/points_raw", PointCloud2, pp.lidar_callback, queue_size=1, buff_size=2 ** 12)

    pubbbox = rospy.Publisher("/det3d/bbox", BoundingBoxArray, queue_size=1)
    # pubPoint = rospy.Publisher("/lidar_points", PointCloud2, queue_size=1)
    print('[+]-------->ROS is starting!')

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
