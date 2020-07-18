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

import torch
import torchvision.transforms as transforms
from models import GeneratorUNet

from sensor_manager import SensorManager, scan_usb
from controller import Controller
from controller.passive_xbox import JoyStick
from local_planner import get_cost_map, get_cmd
from camera_info import camera2lidar
from get_nav import NavMaker

parser = argparse.ArgumentParser()
parser.add_argument('--img_height', type=int, default=128, help='size of image height')
parser.add_argument('--img_width', type=int, default=256, help='size of image width')
parser.add_argument('--show', type=bool, default=False, help='show image')
opt = parser.parse_args()


random.seed(datetime.now())
torch.manual_seed(999)
#torch.cuda.manual_seed(999)
#torch.set_num_threads(12)

#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

generator = GeneratorUNet()

generator = generator.to(device)
generator.load_state_dict(torch.load('ckpt/g.pth', map_location=device))
generator.eval()

img_trans_ = [
    transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
img_trans = transforms.Compose(img_trans_)

def get_nav():
    global nav_maker
    nav = nav_maker.get()
    return nav

def get_img(nav):
    img = sm['camera'].getImage()
    img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    img = img_trans(img)
    nav = img_trans(nav)
    input_img = torch.cat((img, nav), 0).unsqueeze(0)
    return input_img
    

def get_net_result(input_img):
    with torch.no_grad():
        input_img = input_img#.to(device)
        result = generator(input_img)
        return result

theta_y = 20.0*np.pi/180.
pitch_rotationMat = np.array([
    [np.cos(theta_y),  0., np.sin(theta_y)],
    [       0.,        1.,         0.     ],
    [-np.sin(theta_y), 0., np.cos(theta_y)],
])  
    
def inverse_perspective_mapping(img):
    global sm, ctrl, joystick, file
    point_cloud = sm['lidar'].get()
    mask = np.where((point_cloud[3] > 10))[0]
    point_cloud = point_cloud[:,mask][:3,:]
    point_cloud = np.dot(pitch_rotationMat, point_cloud)
    img = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_AREA)
    res = np.where(img > 100)
    image_uv = np.stack([res[1],res[0]])
    trans_pc = camera2lidar(image_uv)
    img = get_cost_map(trans_pc, point_cloud, False)
    file_name = str(time.time())
    cv2.imwrite('record/'+file_name+'.png', img) 
    yaw = get_cmd(img, show=False)
    passive_rotation = 2.4*yaw
    speed, rotation = joystick.get()
    ctrl.set_speed(1.0)
    ctrl.set_rotation(rotation)
    file.write(file_name+'\t'+str(rotation)+'\t'+str(passive_rotation)+'\n')


if __name__ == '__main__':
    ctrl = Controller(scan_usb('CAN'))
    ctrl.start()
    ctrl.set_forward()
    ctrl.set_max_speed(1000)
    
    joystick = JoyStick()
    joystick.start()
    file = open('record.txt', 'a+')
    sensor_dict = {
        'lidar':None,
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
        if False:
            cv2.imshow('Nav', np.array(nav))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        input_img = get_img(nav)
        result = get_net_result(input_img)[0][0]
        result = result.data.numpy()*255+255
        inverse_perspective_mapping(result)
    file.close()
    sm.close_all()