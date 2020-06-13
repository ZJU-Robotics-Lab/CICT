#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2

from gps_reader import read_gps, get_points, get_gps_key
from camera_info import lidar2camera
from pd_filter import filt_pd

data_index = 3
read_gps(data_index)

for i in range(1000, 20000, 20):
    key = get_gps_key(i)
    img = cv2.imread('/home/wang/DataSet/yqdata/images'+str(data_index)+'/'+str(key)+'.png')
    
    point_cloud = get_points(key, num=150)
    point_cloud = filt_pd(point_cloud, False)
    lidar2camera(img, point_cloud,file_name=str(key))