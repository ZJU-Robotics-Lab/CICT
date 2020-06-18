#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 20:05:12 2020

@author: wang
"""
import glob
import numpy as np
#import matplotlib.pyplot as plt

icp_ts_list = []
trans_list = []
rotate_list = []
img_ts_list = []
img_icp_ts_index = []
locations = []

def read_icp_results(file_path):
    global icp_ts_list, trans_list, rotate_list
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            sp_line = line.split()
            ts = float(sp_line[0])
            dx = float(sp_line[1])
            dy = float(sp_line[2])
            dz = float(sp_line[3])
            x = float(sp_line[4])
            y = float(sp_line[5])
            z = float(sp_line[6])
            w = float(sp_line[7])
            
            X, Y, Z = quaternion_to_euler(x, y, z, w)
            
            icp_ts_list.append(ts)
            trans_list.append([dx, dy, dz])
            rotate_list.append([X, Y, Z])
            
    return icp_ts_list, trans_list, rotate_list

def get_locations(data_index=6):
    global locations
    ts_list, trans_list, rotate_list = read_icp_results('/media/wang/DATASET/icp'+str(data_index)+'/trajectory.txt')
    
    x = []
    y = []
    z = []
    location = np.array([[0], [0], [0]])
    for i in range(len(trans_list)):
        theta_x = rotate_list[i][0]
        theta_y = rotate_list[i][1]
        theta_z = rotate_list[i][2]
        
        T = np.array(trans_list[i])
        
        RxMat = np.array([
            [1.0, 0.0, 0.0],
            [0.0,  np.cos(theta_x),  -np.sin(theta_x)],
            [0.0,  np.sin(theta_x),  np.cos(theta_x)],
        ])
        
        RyMat = np.array([
            [np.cos(theta_y),  0.0,  np.sin(theta_y)],
            [0.0,  1.0,  0.0],
            [-np.sin(theta_y),  0.0, np.cos(theta_y)],
        ])
            
        RzMat = np.array([
            [np.cos(theta_z), -np.sin(theta_z), 0.0],
            [np.sin(theta_z),  np.cos(theta_z), 0.0],
            [0.0,  0.0,  1.0],
        ])
            
        R = np.dot(np.dot(RzMat, RyMat), RxMat)
    
        new_location = np.dot(R, location) + np.tile(T, (location.shape[1], 1)).T
        x.append(new_location[0][0])
        y.append(new_location[1][0])
        z.append(new_location[2][0])
        
    locations = np.array([x, y, z]).T
    return locations
    #fig,ax = plt.subplots(figsize=(15, 15))
    #plt.plot(x,y,'xkcd:green',linewidth=4)
    #plt.show()
    
def read_imgs(data_index=6):
    global icp_ts_list, img_ts_list, img_icp_ts_index
    files = glob.glob('/media/wang/DATASET/images'+str(data_index)+'/*.png')
    img_ts_list = []
    for file in files:
        ts = file.split('/')[5][:-4]
        img_ts_list.append(ts)
        
    img_ts_list.sort()
    img_icp_ts_index = []
    for ts in img_ts_list:
        print(img_ts_list.index(ts)/len(img_ts_list))
        bias = [abs(float(item) - float(ts)) for item in icp_ts_list]
        min_bias = min(bias)
        index = bias.index(min_bias)
        img_icp_ts_index.append(index)
    return img_ts_list, img_icp_ts_index

def get_img_path(index):
    global img_ts_list
    assert len(img_ts_list) > 0
    return img_ts_list[index % len(img_ts_list)]
   
def get_points(index, num=300):
    global img_icp_ts_index, locations
    
    icp_index = img_icp_ts_index[index]
    
    dy = locations[icp_index+5][1] - locations[max(0, icp_index-5)][1]
    dx = locations[icp_index+5][0] - locations[max(0, icp_index-5)][0]
    angle = np.arctan2(dy, dx)
    
    x = []
    y = []
    z = []
    start = locations[icp_index]
    for i in range(1,num):
        data = locations[icp_index+i]
        _x = (data[0]-start[0])
        _y = data[1]-start[1]
        x.append(_x*np.cos(angle) + _y*np.sin(angle))
        y.append(_y*np.cos(angle) - _x*np.sin(angle))
        z.append(-1.8)

    return np.array([x, y, z])
            

def quaternion_to_euler(x, y, z, w):

        import math
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        X = math.degrees(math.atan2(t0, t1))

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        Y = math.degrees(math.asin(t2))

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        Z = math.degrees(math.atan2(t3, t4))

        return X, Y, Z
    
if __name__ == '__main__':
    get_locations()