#! /home/bang/anaconda3/envs/openmmlab/bin/python

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
import ros_numpy



class Detect3D:
    def __init__(self, config_path, checkpoint, device):
        self.config = config_path
        self.checkpoint = checkpoint
        self.device = device
        self.model = init_model(config_path, checkpoint, device)
        print('[+]-------->Model has been inited!')

        self.subLidar = rospy.Subscriber("/velodyne_points", PointCloud2, self.lidar_callback,tcp_nodelay=True)
        self.pubbbox = rospy.Publisher("/obj", BoundingBoxArray, queue_size=1)
        print('[+]-------->ROS is starting!')


    def lidar_callback(self, PointCloudsmsg):
        callback_start =  time.time()

        # method 1 : 直接使用 read_points_list 慢的吓人
        # points_array = np.array(pc2.read_points_list(PointCloudsmsg, field_names=("x", "y", "z", "intensity"), skip_nans=True)) 
        # points_array = points_array.astype('float32')

        # method 2 : 也不知道为啥，反正在分两步转换成list再转换成array就会快很多
        # points_array = np.array(list(pc2.read_points(PointCloudsmsg, field_names=("x", "y", "z", "intensity"), skip_nans=True))) 
        # points_array = points_array.astype('float32')

        # method 3 : 这样是最快的，贼tm快，比 pc2.read_points 耗时小一两个数量级
        # pc = ros_numpy.numpify(PointCloudsmsg)
        # points=np.zeros((pc.shape[0],4))
        # points[:,0]=pc['x']
        # points[:,1]=pc['y']
        # points[:,2]=pc['z']
        # points[:,3]=pc['intensity']
        # points_array = np.array(points, dtype=np.float32)

        pc = ros_numpy.numpify(PointCloudsmsg)
        points=np.zeros((pc.shape[0],5))
        points[:,0]=pc['x']
        points[:,1]=pc['y']
        points[:,2]=pc['z']
        points[:,3]=pc['intensity']
        points[:,4]=pc['ring']

        points_array = np.array(points, dtype=np.float32)

        inference_start = time.time()
        
        # 开始推理，结果保存在result中
        torch.cuda.synchronize()
        result, data = inference_detector(self.model, points_array)
        torch.cuda.synchronize()

        inference_end = time.time()

        print('single frame use:', (inference_start - inference_end)) # 看一下单帧推理时间

        self.PubBBox(result,PointCloudsmsg)

        callback_end = time.time()
        print('callback function used total_time:', (callback_start - callback_end)) # 打印整个回调函数总共的时间

    def PubBBox(self, result, PointCloudsmsg):
        scores = result.pred_instances_3d.scores_3d.cpu().numpy()
        mask = scores > 0.6
        scores = scores[mask]
        boxes_lidar = result.pred_instances_3d.bboxes_3d[mask].cpu().numpy()
        
        if boxes_lidar.shape[0] != 0:
            print(f"Detected {boxes_lidar.shape[0]} boundboxs ")
        label = result.pred_instances_3d.labels_3d[mask].cpu().numpy()

        # 创建一个BoundingBoxArray
        bboxarr = BoundingBoxArray()
        # BoundingBoxArray的参考坐标系就是激光雷达坐标系
        bboxarr.header = PointCloudsmsg.header
        # 将result中所有的BoundingBox塞到BoundingBoxArray里面
        for i in range(boxes_lidar.shape[0]):
            # 创建一个BoundingBox
            bbox = BoundingBox()
            bbox.header = PointCloudsmsg.header
            bbox.pose.position.x = float(boxes_lidar[i][0])
            bbox.pose.position.y = float(boxes_lidar[i][1])
            bbox.pose.position.z = float(boxes_lidar[i][2])+float(boxes_lidar[i][5])/2
            bbox.dimensions.x = float(boxes_lidar[i][3])
            bbox.dimensions.y = float(boxes_lidar[i][4])
            bbox.dimensions.z = float(boxes_lidar[i][5])
            # q = Quaternion(axis=(0, 0, 1), radians=float(boxes_lidar[i][6])) # 通过yaw角创建四元数 kitti
            q = Quaternion(axis=(0, 0, 1), radians=float(-boxes_lidar[i][6]) + 0.5*np.pi) # 通过yaw角创建四元数 nuscenes

            bbox.pose.orientation.x = q.x
            bbox.pose.orientation.y = q.y
            bbox.pose.orientation.z = q.z
            bbox.pose.orientation.w = q.w
            bbox.value = scores[i]
            bbox.label = label[i]
            bboxarr.boxes.append(bbox)
        # 发布BoundingBoxArray
        self.pubbbox.publish(bboxarr)

if __name__ == '__main__':

    PATH = '/home/bang/mmdetection3d/'

    # pointpillars
    # config_path = PATH + 'pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py'
    # checkpoints = PATH + 'hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth'

    config_path = PATH + 'configs/pointpillars/pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d.py'
    checkpoints = PATH + '/models/hv_pointpillars_secfpn_sbn-all_fp16_2x8_2x_nus-3d_20201020_222626-c3f0483e.pth'

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

    OLIO_detect = Detect3D(config_path, checkpoints, device)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
