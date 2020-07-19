#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import cv2
import time
import random
import argparse
import numpy as np
from PIL import Image
from datetime import datetime

from sensor_manager import SensorManager, scan_usb
from local_planner import get_cost_map, get_cmd
from camera_info import camera2lidar
from get_nav import NavMaker

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
    #img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    #img = img_trans(img)
    #nav = img_trans(nav)
    #input_img = torch.cat((img, nav), 0).unsqueeze(0)
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