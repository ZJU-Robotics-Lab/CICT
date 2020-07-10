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

from sensor_manager import SensorManager
from controller import Controller
from local_planner import get_cost_map, get_cmd, read_pcd
from camera_info import camera2lidar

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs of training')
parser.add_argument('--dataset_name', type=str, default="test03", help='name of the dataset')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch from which to start lr decay')
parser.add_argument('--n_cpu', type=int, default=16, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_height', type=int, default=128, help='size of image height')
parser.add_argument('--img_width', type=int, default=256, help='size of image width')
parser.add_argument('--channels', type=int, default=3, help='number of image channels') 
opt = parser.parse_args()


random.seed(datetime.now())
torch.manual_seed(999)
torch.cuda.manual_seed(999)
torch.set_num_threads(16)

#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

generator = GeneratorUNet()

generator = generator.to(device)
generator.load_state_dict(torch.load('ckpt/g.pth'))
generator.eval()

img_trans_ = [
    transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
img_trans = transforms.Compose(img_trans_)

def get_img():
    #img = sm['camera'].getImage()
    img = cv2.imread("/media/wang/DATASET/images6/1592380358.292274.png")
    img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

    # TODO
    nav = cv2.imread("/media/wang/DATASET/nav6/1592380358.292274.png")
    nav = Image.fromarray(cv2.cvtColor(nav,cv2.COLOR_BGR2RGB))
    
    img = img_trans(img)
    nav = img_trans(nav)
    
    input_img = torch.cat((img, nav), 0).unsqueeze(0)
    return input_img
    

def get_net_result(input_img):
    with torch.no_grad():
        input_img = input_img.to(device)
        result = generator(input_img)
        return result

theta_y = 20.0*np.pi/180.
pitch_rotationMat = np.array([
    [np.cos(theta_y),  0., np.sin(theta_y)],
    [       0.,        1.,         0.     ],
    [-np.sin(theta_y), 0., np.cos(theta_y)],
])  
    
def inverse_perspective_mapping(img):
    #29ms
    point_cloud, intensity = read_pcd('/media/wang/DATASET/pcd6/1592380357.256119251.pcd')
    point_cloud = np.dot(pitch_rotationMat, point_cloud)
    #2.2 ms
    img = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_AREA)
    res = np.where(img > 100)
    image_uv = np.stack([res[1],res[0]])
    #1.9 ms
    trans_pc = camera2lidar(image_uv)
    #2.2 ms
    img = get_cost_map(trans_pc, point_cloud, False)
    #2.1 ms
    v, w = get_cmd(img, show=False)
    #print(v, w)
    

if __name__ == '__main__':
    """
    sensor_dict = {'camera':None,
                   'lidar':None,
                   }
    sm = SensorManager(sensor_dict)
    sm.init_all()

    ctrl = Controller()
    ctrl.start()
    ctrl.set_forward()
    """
    while True:
        input_img = get_img()
        t1 = time.time()
        result = get_net_result(input_img)[0][0]
        result = result.cpu().data.numpy()*255+255
        #img_gray = cv2.resize(result,(256*4, 128*4),interpolation=cv2.INTER_CUBIC)
        #cv2.imshow("img_gray",img_gray)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        t2 = time.time()
        print('get_net_result', round(1000*(t2-t1),3), 'ms')
        inverse_perspective_mapping(result)
        t3 = time.time()
        print('inverse_perspective_mapping', round(1000*(t3-t2)-29,3), 'ms')
        break