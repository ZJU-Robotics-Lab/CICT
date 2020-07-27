#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np

width = 1280
height = 720

fov = 90.
fx = 1280/(2*np.tan(fov*np.pi/360))#$711.642238
fy = fx#711.302135
s = 0.0
x0 = 1280/2#644.942373
y0 = 720/2#336.030580


cameraMat = np.array([
        [fx,  s, x0],
        [0., fy, y0],
        [0., 0., 1.]
])

theta_y = 0.0*np.pi/180.

pitch_rotationMat = np.array([
    [np.cos(theta_y),  0., np.sin(theta_y)],
    [       0.,        1.,         0.     ],
    [-np.sin(theta_y), 0., np.cos(theta_y)],
])


translationMat = np.array([0.0, 0.0, 0.0])

theta_x = 0.
theta_y = 0.
theta_z = 0.

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
    
rotationMat = np.dot(np.dot(RzMat, RyMat), RxMat)
#rotationMat = np.dot(rotationMat, np.linalg.inv(pitch_rotationMat))


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
    cv2.imwrite('/media/wang/DATASET/label'+str(data_index)+'/'+file_name+'.png',img)
    
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
    
def camera2lidar(image_uv):
    rotation = np.linalg.inv(np.dot(cameraMat, rotationMat))
    translation = np.dot(cameraMat, translationMat)
    translation = np.dot(rotation, translation)
    #R = rotation
    T = translation
    roadheight = -1.95
    
    u = image_uv[0]
    v = image_uv[1]

    R = np.array([[-4.06959738e-06,-1.40149986e-04,1.04474300e+00],
                 [-1.40517726e-03,4.37472699e-06,9.02148828e-01],
                 [-3.97016293e-06,-1.39887464e-03,3.72919730e-01],])
    
    zi = (T[2]+roadheight)/(R[2][0]*u + R[2][1]*v + R[2][2])
    xl = (R[0][0]*u + R[0][1]*v + R[0][2])*zi - T[0]
    yl = (R[1][0]*u + R[1][1]*v + R[1][2])*zi - T[1]

    trans_pc = np.array([
            xl,
            yl,
            [roadheight]*image_uv.shape[1]
            ])

    return trans_pc