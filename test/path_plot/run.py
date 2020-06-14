#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2

from gps_reader import read_gps, get_points, get_gps_key
from camera_info import lidar2camera
from pd_filter import filt_pd

data_index = 1
read_gps(data_index)

for i in range(0, 20000, 20):
    key = get_gps_key(i)
    file_name = str(key)
    if len(file_name) < 17:
        file_name = file_name + '0'*(17-len(file_name))
    img = cv2.imread('/home/wang/DataSet/yqdata/images'+str(data_index)+'/'+file_name+'.png')
    point_cloud = get_points(key, num=250)
    point_cloud = filt_pd(point_cloud, False)
    lidar2camera(img, point_cloud,file_name=str(key))