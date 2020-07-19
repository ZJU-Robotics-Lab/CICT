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
from imitation_model import Model

from sensor_manager import SensorManager, scan_usb
from controller import Controller
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
    #t1 = time.time()
    point_cloud = sm['lidar'].get()
    mask = np.where((point_cloud[3] > 10))[0]
    point_cloud = point_cloud[:,mask][:3,:]
    point_cloud = np.dot(pitch_rotationMat, point_cloud)
    #t2 = time.time()
    #2.2 ms
    img = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_AREA)
    res = np.where(img > 100)
    image_uv = np.stack([res[1],res[0]])
    #t3 = time.time()
    #1.9 ms
    trans_pc = camera2lidar(image_uv)
    #2.2 ms
    img = get_cost_map(trans_pc, point_cloud, False)
    #2.1 ms
    cost_map = transform(img)
    output = cmd_model(cost_map.unsqueeze(0))
    cmd = output.data.numpy()[0][0]
    #yaw = get_cmd(img, show=opt.show)

    #direct = 1.0 if w >0 else -1.0
    #rotation = np.rad2deg(yaw)/30.
    #print(direct*2.4*abs(yaw))
    ctrl.set_speed(1.0)
    #ctrl.set_rotation(2.4*yaw)
    ctrl.set_rotation(cmd)
    #ctrl.set_rotation(direct*2.4*abs(yaw))
    
    #t4 = time.time()
    #print('get pcd', round(1000*(t2-t1),3), 'ms')
    #print('process img', round(1000*(t3-t2),3), 'ms')
    #print('get cmd', round(1000*(t4-t3),3), 'ms')

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
        #t1 = time.time()
        x,y,t = sm['gps'].get()
        nav = get_nav()
        if False:
            cv2.imshow('Nav', np.array(nav))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        input_img = get_img(nav)
        #t2 = time.time()
        #t1 = time.time()
        result = get_net_result(input_img)[0][0]
        result = result.data.numpy()*255+255
        #img_gray = cv2.resize(result,(256*4, 128*4),interpolation=cv2.INTER_CUBIC)
        #cv2.imshow("img_gray",img_gray)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        #t3 = time.time()
        inverse_perspective_mapping(result)
        #t4= time.time()
        #print('img process', round(1000*(t2-t1),3), 'ms')
        #print('get_net_result', round(1000*(t3-t2),3), 'ms')
        #print('inverse_perspective_mapping', round(1000*(t4-t3),3), 'ms')
        #print('total', round(1000*(t4-t1),3), 'ms')
    sm.close_all()