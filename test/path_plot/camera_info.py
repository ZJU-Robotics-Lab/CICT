#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np

width = 1280
height = 720

fx = 711.642238
fy = 711.302135
s = 0.0
x0 = 644.942373
y0 = 336.030580

cameraMat = np.array([
        [fx,  s, x0],
        [0., fy, y0],
        [0., 0., 1.]
])

distortionMat = np.array([-0.347125, 0.156284, 0.001037, -0.000109 ,0.000000])

theta_y = 10.0*np.pi/180.

pitch_rotationMat = np.array([
    [np.cos(theta_y),  0., np.sin(theta_y)],
    [       0.,        1.,         0.     ],
    [-np.sin(theta_y), 0., np.cos(theta_y)],
])


rotationMat = np.array([
    [-0.0024, -1.0000, -0.0033],
    [0.0746,  0.0031,  -0.9972],
    [0.9972,  -0.0026, 0.0746],
])
translationMat = np.array([0.0660, 0.1263, 0.2481])

theta_x = np.arctan2(rotationMat[2][1], rotationMat[2][2])

"""
theta_y = np.arctan2(-rotationMat[2][0], np.sqrt(rotationMat[2][1]**2 + rotationMat[2][2]**2))
theta_z = np.arctan2(rotationMat[1][0], rotationMat[0][0])

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
print(R, rotationMat)
print(theta_x*180./np.pi, theta_y*180./np.pi, theta_z*180./np.pi)
"""
rotationMat = np.dot(rotationMat, np.linalg.inv(pitch_rotationMat))


def lidar2camera(point_cloud, rotationMat=rotationMat, translationMat=translationMat, file_name='merge', data_index=1):
    img = np.zeros((720, 1280, 3), np.uint8)
    trans_pc = np.dot(rotationMat, point_cloud) + np.tile(translationMat, (point_cloud.shape[1], 1)).T
    image_uv = np.array([
            trans_pc[0]*fx/trans_pc[2] + x0,
            trans_pc[1]*fy/trans_pc[2] + y0
            ])
    total = image_uv.shape[1]
    for i in range(total):
        point = (int(image_uv[0][i]), int(image_uv[1][i]))
        if point[0] > width or point[0] < 0 or point[1] > height or point[1] < 0:
            continue
        #cv2.circle(img, point, 2, (i/total*255, 0, 255-i/total*255), 8)
        cv2.circle(img, point, 2, (255, 255, 255), 8)
    cv2.imwrite('output'+str(data_index)+'/'+file_name+'.png',img)
    
def lidar2camera_test(img, point_cloud, rotationMat=rotationMat, translationMat=translationMat, file_name='merge', data_index=1):
    trans_pc = np.dot(rotationMat, point_cloud) + np.tile(translationMat, (point_cloud.shape[1], 1)).T
    image_uv = np.array([
            trans_pc[0]*fx/trans_pc[2] + x0,
            trans_pc[1]*fy/trans_pc[2] + y0
            ])
    total = image_uv.shape[1]
    for i in range(total):
        point = (int(image_uv[0][i]), int(image_uv[1][i]))
        if point[0] > width or point[0] < 0 or point[1] > height or point[1] < 0:
            continue
        cv2.circle(img, point, 2, (i/total*255, 0, 255-i/total*255), 8)
    cv2.imwrite('test_output/test_'+file_name+'.png',img)