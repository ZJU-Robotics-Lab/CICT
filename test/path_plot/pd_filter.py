#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from manual_gps import dist_p2p

def rm_noise(point_cloud):
    x = []
    y = []
    z = []
    dists = [999]
    total_len = len(point_cloud[0])
    for i in range(total_len-1):
        dist = dist_p2p(point_cloud[0][i], point_cloud[1][i],point_cloud[0][i+1], point_cloud[1][i+1])
        if abs(dist) < 0.00001:
            pass
        else:
            if dist > 3 and dists[-1] > 3:
                continue
            dists.append(dist)
            x.append(point_cloud[0][i])
            y.append(point_cloud[1][i])
            z.append(point_cloud[2][i])
            
    return np.array([x, y, z])

def filt_pd(point_cloud, show=False):
    point_cloud = rm_noise(point_cloud)
    
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
    
    if show:
        fig,ax = plt.subplots(figsize=(15, 15))
        plt.scatter(point_cloud[0],fit_data)
        #plt.scatter(point_cloud[0],point_cloud[1])
        plt.plot(point_cloud[0],point_cloud[1],'xkcd:green',linewidth=4)
        plt.show()
    
    point_cloud = np.array([point_cloud[0], fit_data, point_cloud[2]])
    # rotate point_cloud
    point_cloud = np.dot(np.linalg.inv(rot_mat), point_cloud)
    point_cloud = point_cloud[:,:-50]
    point_cloud = fill_points(point_cloud)
    return point_cloud

def fill_points(point_cloud):
    min_dist = 0.03
    path_width = 1.0
    pc_x = [0]
    pc_y = [0]
    pc_z = [-1.8]
    for i in range(len(point_cloud[0])-1):
        x = point_cloud[0][i]
        y = point_cloud[1][i]
        z = point_cloud[2][i]
        next_x = point_cloud[0][i+1]
        next_y = point_cloud[1][i+1]
        
        dist = dist_p2p(x, y, next_x, next_y)
        if dist > 15.0:
            print('Error: distance too large', dist)
            return point_cloud
        if dist > min_dist:
            n = int(dist/min_dist)
            for j in range(n):
                pc_x.append(x + (next_x - x)*j/n)
                pc_y.append(y + (next_y - y)*j/n)
                pc_z.append(z)
        else:
            pc_x.append(x)
            pc_y.append(y)
            pc_z.append(z)
    
    pc_num = len(pc_x)
    for i in range(pc_num-1):
        dy = pc_y[i+1] - pc_y[i]
        dx = pc_x[i+1] - pc_x[i]
        angle = np.arctan2(dy, dx)
        
        n = int(path_width/2/min_dist)
        for j in range(n):
            x_left = pc_x[i] + np.cos(angle + np.pi/2)*j/n*path_width/2
            y_left = pc_y[i] + np.sin(angle + np.pi/2)*j/n*path_width/2
            x_right = pc_x[i] + np.cos(angle - np.pi/2)*j/n*path_width/2
            y_right = pc_y[i] + np.sin(angle - np.pi/2)*j/n*path_width/2
            pc_x.append(x_left)
            pc_y.append(y_left)
            pc_x.append(x_right)
            pc_y.append(y_right)
            pc_z.append(pc_z[i])
            pc_z.append(pc_z[i])
        
    return np.array([pc_x, pc_y, pc_z])