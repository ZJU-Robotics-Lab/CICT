#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
#from gps_reader import read_gps, get_points, get_gps_key
from camera_info import lidar2camera,lidar2camera_test
from icp_reader import get_locations,read_imgs, get_points, get_img_path
from pd_filter import filt_pd

"""
for data_index in [3]:
    #data_index = 1
    read_gps(data_index)
    try:
        for i in range(3000, 20000, 100):
            key = get_gps_key(i)
            #file_name = str(key)
            #if len(file_name) < 17:
            #    file_name = file_name + '0'*(17-len(file_name))
            img = cv2.imread('/home/wang/DataSet/yqdata/images'+str(data_index)+'/'+key+'.png')
            point_cloud = get_points(key, num=250)
            point_cloud = filt_pd(point_cloud, False)
            #lidar2camera(point_cloud,file_name=str(key), data_index=data_index)
            lidar2camera_test(img, point_cloud,file_name=str(key), data_index=data_index)
    except:
        pass
        
"""

for data_index in [6]:
    get_locations(data_index)
    read_imgs(data_index)
    for index in range(1, 20000, 100):
        img_path = get_img_path(index)
        point_cloud = get_points(index, num=100)
        point_cloud = filt_pd(point_cloud, False)
        img = cv2.imread('/media/wang/DATASET/images'+str(data_index)+'/'+img_path+'.png')
        lidar2camera_test(img, point_cloud,file_name=img_path, data_index=data_index)