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

from local_planner import get_cost_map, get_cmd, read_pcd
from camera_info import camera2lidar

parser = argparse.ArgumentParser()
parser.add_argument('--img_height', type=int, default=128, help='size of image height')
parser.add_argument('--img_width', type=int, default=256, help='size of image width')

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

def get_img():
    img = cv2.imread("img.png")
    img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

    nav = cv2.imread("nav2.png")
    nav = Image.fromarray(cv2.cvtColor(nav,cv2.COLOR_BGR2RGB))
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
    point_cloud, intensity = read_pcd('pcd.pcd')
    intensity = np.array(intensity)
    mask = np.where((intensity > 10))[0]
    point_cloud = point_cloud[:,mask]
    point_cloud = np.dot(pitch_rotationMat, point_cloud)
    img = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_AREA)
    res = np.where(img > 100)
    image_uv = np.stack([res[1],res[0]])
    trans_pc = camera2lidar(image_uv)
    img = get_cost_map(trans_pc, point_cloud, False)
    v, w = get_cmd(img, show=True)
    #print(v, w)

if __name__ == '__main__':
    while True:
        input_img = get_img()
        result = get_net_result(input_img)[0][0]
        result = result.data.numpy()*255+255
        inverse_perspective_mapping(result)