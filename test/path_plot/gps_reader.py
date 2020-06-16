#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import glob
import numpy as np
import matplotlib.pyplot as plt
from pd_filter import rm_noise
from manual_gps import find_nn

gps_info = {}
keys = []

def read_gps(data_index=5):
    global gps_info, keys
    files = glob.glob('/home/wang/DataSet/yqdata/images'+str(data_index)+'/*.png')
    file_path = []
    for file in files:
        #ts = float(file.split('/')[6][:-4])
        ts = file.split('/')[6][:-4]
        file_path.append(ts)
        
    file_path.sort()

    with open('/home/wang/DataSet/yqdata/gps'+str(data_index)+'/gps.txt', 'r') as file:
        lines = file.readlines()
        for line in lines:
            try:
                sp_line = line.split()
                #ts = float(sp_line[0])
                ts = sp_line[0]
                data = sp_line[2][1:-3].split('\\t')
                x = float(data[0])
                y = float(data[1])
                gps_info[ts] = [x, y]
            except:
                pass
    
    keys = list(gps_info.keys())
    for item in file_path:
        if item not in keys:
            print('Error', item)


def get_points(key, num=300):
    global gps_info, keys
    assert len(keys) > 0
    angle = get_filt_angle(key)
    
    index = keys.index(key)
    x = []
    y = []
    z = []
    start = gps_info[keys[index]]
    for i in range(1,num):
        data = gps_info[keys[index+i]]
        _x = (data[0]-start[0])
        _y = data[1]-start[1]
        x.append(_x*np.cos(angle) + _y*np.sin(angle))
        y.append(-_y*np.cos(angle) + _x*np.sin(angle))
        z.append(-1.8)

    return np.array([x, y, z])

def get_angle(key):
    global gps_info, keys
    frame_index = keys.index(key)
    dy = gps_info[keys[frame_index+15]][1] - gps_info[keys[frame_index]][1]
    dx = gps_info[keys[frame_index+15]][0] - gps_info[keys[frame_index]][0]
    angle = np.arctan2(dy, dx)
    return angle

def get_filt_angle(key, show=False):
    global gps_info, keys
    index = keys.index(key)
    
    x = []
    y = []
    z = []
    start = gps_info[key]
    for i in range(200):
        data = gps_info[keys[max(0, index+i-100)]]
        _x = data[0]-start[0]
        _y = data[1]-start[1]
        x.append(_x)
        y.append(_y)
        z.append(-1.8)
        
    point_cloud = np.array([x, y, z])

    chosen_point = [0, 0]
    head_x = point_cloud[0][0]
    head_y = point_cloud[1][0]
    end_x = point_cloud[0][-1]
    end_y = point_cloud[1][-1]
    rot = np.arctan2((end_y - head_y), (end_x - head_x))
    rot_mat = np.array([
        [np.cos(rot), np.sin(rot), 0.0],
        [-np.sin(rot), np.cos(rot), 0.0],
        [0.0,  0.0,  1.0],
    ])
    # rotate point_cloud
    point_cloud = np.dot(rot_mat, point_cloud)
    
    fit_param = np.polyfit(point_cloud[0], point_cloud[1], 4)
    poly_func = np.poly1d(fit_param)
    fit_data = poly_func(point_cloud[0])
   
    point_cloud = np.array([point_cloud[0], fit_data, point_cloud[2]])
    # rotate point_cloud
    point_cloud = np.dot(np.linalg.inv(rot_mat), point_cloud)
    
    
    nn_x, nn_y, nn_index = find_nn(chosen_point[0], chosen_point[1], point_cloud[0], point_cloud[1])
    
    if show:
        fig,ax = plt.subplots(figsize=(15, 15))
        plt.scatter(point_cloud[0],point_cloud[1])
        plt.plot([chosen_point[0], nn_x],[chosen_point[1], nn_y],'xkcd:red',linewidth=4)
        plt.show()
        
    dy = point_cloud[1][min(len(point_cloud[0])-1, nn_index+1)] - point_cloud[1][nn_index-1]
    dx = point_cloud[0][min(len(point_cloud[0])-1, nn_index+1)] - point_cloud[0][nn_index-1]
    angle = np.arctan2(dy, dx)
    return angle

def get_gps_key(index):
    global keys
    assert len(keys) > 0
    return keys[index % len(keys)]