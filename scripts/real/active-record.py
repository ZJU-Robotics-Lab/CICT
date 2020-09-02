#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from os.path import join, dirname
sys.path.insert(0, join(dirname(__file__), '..'))

import cv2
import time
import random
import argparse
import numpy as np
from PIL import Image
from datetime import datetime

import torch
import torchvision.transforms as transforms
from learning.models import GeneratorUNet

from device import SensorManager, scan_usb
from device.controller import Controller
from utils.local_planner import get_cost_map, get_cmd
from utils.camera_info import camera2lidar
from utils.navigator import NavMaker

parser = argparse.ArgumentParser()
parser.add_argument('--img_height', type=int, default=128, help='size of image height')
parser.add_argument('--img_width', type=int, default=256, help='size of image width')
parser.add_argument('--show', type=bool, default=False, help='show image')
opt = parser.parse_args()

random.seed(datetime.now())
torch.manual_seed(999)

device = torch.device('cpu')

generator = GeneratorUNet()

generator = generator.to(device)
generator.load_state_dict(torch.load('../ckpt/g.pth', map_location=device))
generator.eval()

img_trans_ = [
    transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
img_trans = transforms.Compose(img_trans_)

global_nav = None
global_img = None
global_res = None
def get_nav():
    global nav_maker, global_nav
    nav = nav_maker.get()
    global_nav = cv2.cvtColor(np.asarray(nav),cv2.COLOR_RGB2BGR) 
    return nav

def get_img(nav):
    global global_img
    img = sm['camera'].getImage()
    global_img = img
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
    global sm, ctrl, file
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
    cv2.imwrite('record/'+file_name+'_nav.png', global_nav) 
    cv2.imwrite('record/'+file_name+'_img.png', global_img) 
    cv2.imwrite('record/'+file_name+'_res.png', global_res) 
    yaw = get_cmd(img, show=False)
    rotation = -2.4*yaw
    # speed, rotation = joystick.get()
    ctrl.set_speed(1.0)
    ctrl.set_rotation(rotation)
    # print(-rotation, passive_rotation)
    file.write(file_name+'\t'+str(-rotation)+'\n')


if __name__ == '__main__':
    ctrl = Controller(scan_usb('CAN'))
    ctrl.start()
    ctrl.set_forward()
    ctrl.set_max_speed(1000)
    
    # joystick = JoyStick()
    # joystick.start()
    file = open('record/record.txt', 'a+')
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
        input_img = get_img(nav)
        result = get_net_result(input_img)[0][0]
        result = result.data.numpy()*255+255
        global_res = cv2.resize(result,(256, 128),interpolation=cv2.INTER_CUBIC)
        inverse_perspective_mapping(result)
    file.close()
    sm.close_all()