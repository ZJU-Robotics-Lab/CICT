#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from os.path import join, dirname
sys.path.insert(0, join(dirname(__file__), '..'))

import cv2
import time
import argparse
import numpy as np

from device import SensorManager
from utils.navigator import NavMaker

parser = argparse.ArgumentParser()
parser.add_argument('--img_height', type=int, default=128, help='size of image height')
parser.add_argument('--img_width', type=int, default=256, help='size of image width')
parser.add_argument('--show', type=bool, default=False, help='show image')
opt = parser.parse_args()

def get_nav():
    nav = nav_maker.get()
    nav = cv2.cvtColor(np.asarray(nav),cv2.COLOR_RGB2BGR) 
    return nav

def get_img(nav):
    img = sm['camera'].getImage()
    return img

if __name__ == '__main__':
    file = open('record/record.txt', 'a+')
    sensor_dict = {
        #'lidar':None,
        'camera':None,
        'gps':None,
        'imu':None,
        }
    sm = SensorManager(sensor_dict)
    sm.init_all()
    nav_maker = NavMaker(sm['gps'], sm['imu'])
    nav_maker.start()
    time.sleep(1)
    while True:
        x,y,t = sm['gps'].get()
        nav = get_nav()
        ax, ay, az, yaw, pitch, roll, w = sm['imu'].get()
        img = get_img(nav)
        file_name = str(t)
        cv2.imwrite('record/'+file_name+'.png', img) 
        cv2.imwrite('record/'+file_name+'_nav.png', nav) 
        file.write(file_name+'\t'+str(yaw)+'\t'+str(x)+'\t'+str(y)+'\n')
    file.close()
    sm.close_all()