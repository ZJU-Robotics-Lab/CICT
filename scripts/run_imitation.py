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
from models import GeneratorUNet
from imitation_model import Model

from device import SensorManager, scan_usb
from device.controller import Controller
from utils.local_planner import get_cost_map
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

transforms_ = [
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5)),
]
transform = transforms.Compose(transforms_)
cmd_model = Model().to(device)
cmd_model.load_state_dict(torch.load('ckpt/model.pth', map_location=device))
cmd_model.eval()

def inverse_perspective_mapping(img):
    global sm, ctrl
    point_cloud = sm['lidar'].get()
    mask = np.where((point_cloud[3] > 10))[0]
    point_cloud = point_cloud[:,mask][:3,:]
    point_cloud = np.dot(pitch_rotationMat, point_cloud)
    img = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_AREA)
    res = np.where(img > 100)
    image_uv = np.stack([res[1],res[0]])
    trans_pc = camera2lidar(image_uv)
    img = get_cost_map(trans_pc, point_cloud, False)
    cost_map = transform(img)
    output = cmd_model(cost_map.unsqueeze(0))
    cmd = output.data.numpy()[0][0]
    ctrl.set_speed(1.0)
    print(cmd)
    ctrl.set_rotation(cmd)

if __name__ == '__main__':
    ctrl = Controller(scan_usb('CAN'))
    ctrl.start()
    ctrl.set_forward()
    ctrl.set_max_speed(1000)
    

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

    sm.close_all()