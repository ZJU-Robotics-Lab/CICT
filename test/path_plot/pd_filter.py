#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from manual_gps import dist_p2p

def rm_noise(point_cloud):
    x = []
    y = []
    z = []
    dists = [0]
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

def filt_pd(point_cloud, show=True):
    #rm_noise(point_cloud)
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
    return point_cloud