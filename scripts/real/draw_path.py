#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import glob
import os
from os.path import join, dirname
sys.path.insert(0, join(dirname(__file__), '..'))
sys.path.insert(0, join(dirname(__file__), '../..'))

import cv2
import time
import random
import argparse
import numpy as np
from PIL import Image
from datetime import datetime

import torch
from torch.autograd import grad
import torchvision.transforms as transforms
from learning.models import GeneratorUNet
from learning.path_model import ModelGRU

from utils.local_planner import get_cost_map, project
from utils.camera_info import camera2lidar, lidar2camera

parser = argparse.ArgumentParser()
parser.add_argument('--img_height', type=int, default=128, help='size of image height')
parser.add_argument('--img_width', type=int, default=256, help='size of image width')
opt = parser.parse_args()

random.seed(datetime.now())
torch.manual_seed(999)

# device = torch.device('cpu')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
generator = GeneratorUNet()

generator = generator.to(device)
generator.load_state_dict(torch.load('../../ckpt/g.pth', map_location=device))
generator.eval()
model = ModelGRU().to(device)
model.load_state_dict(torch.load('../../ckpt/model_gru.pth', map_location=device))
model.eval()

img_trans_ = [
    transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
img_trans = transforms.Compose(img_trans_)

costmap_transforms = [
    transforms.Resize((200, 400), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
]
costmap_transforms = transforms.Compose(costmap_transforms)

def get_img(img, nav):
    # img = cv2.imread("/home/wang/video/data5/img/1602576026.821914.png")
    #img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    img = Image.fromarray(img)

    # nav = cv2.imread("/home/wang/video/data5/nav/1602576026.812827.png")
    # nav = Image.fromarray(cv2.cvtColor(nav,cv2.COLOR_BGR2RGB))
    nav = Image.fromarray(nav)
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

def inverse_perspective_mapping(img, pcd):
    # pcd = np.load('/home/wang/video/data5/lidar/1602576026.9392703.npy')

    point_cloud = pcd[:3,:]
    intensity = pcd[3:,:]
    intensity = np.array(intensity)
    mask = np.where((intensity > 10))[0]
    point_cloud = point_cloud[:,mask]
    point_cloud = np.dot(pitch_rotationMat, point_cloud)
    img = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_AREA)
    res = np.where(img > 100)
    image_uv = np.stack([res[1],res[0]])
    trans_pc = camera2lidar(image_uv)
    img = get_cost_map(trans_pc, point_cloud, False)

    # yaw = get_cmd(img, show=False)


def read_files(index):
    file_path = '/home/wang/video/data'+str(index)
    img_list = []
    pcd_list = []
    nav_list = []
    cost_list = []
    for file_name in glob.glob(file_path+'/img/'+'*.png'):
        img_list.append(file_name.split('/')[-1][:-4])
    for file_name in glob.glob(file_path+'/lidar/'+'*.npy'):
        pcd_list.append(file_name.split('/')[-1][:-4])
    for file_name in glob.glob(file_path+'/nav/'+'*.png'):
        nav_list.append(file_name.split('/')[-1][:-4])
    for file_name in glob.glob(file_path+'/cost/'+'*.png'):
        cost_list.append(file_name.split('/')[-1][:-4])
    img_list.sort(), pcd_list.sort(), nav_list.sort(), cost_list.sort()
    return img_list, pcd_list, nav_list, cost_list



def find_nn(ts, ts_list, back=0):
    dt_list = list(map(lambda x: abs(float(x)-float(ts)), ts_list))
    index = max(0, dt_list.index(min(dt_list)) - back)
    return ts_list[index]

if __name__ == '__main__':
    dataset = {}
    for index in [0,1,2,3,4,5]:
        img_list, pcd_list, nav_list, cost_list = read_files(index)
        dataset[index] = {'img_list':img_list, 'pcd_list':pcd_list, 'nav_list':nav_list, 'cost_list':cost_list}

    for index in [0,1,2,3,4,5]:
        choose_dataset = dataset[index]
        for ts in choose_dataset['img_list']:
            img = cv2.imread('/home/wang/video/data'+str(index)+'/img/'+ts+'.png')
            nav_ts = find_nn(ts, choose_dataset['nav_list'])
            pcd_ts = find_nn(ts, choose_dataset['pcd_list'])
            nav = cv2.imread('/home/wang/video/data'+str(index)+'/nav/'+nav_ts+'.png')
            pcd = np.load('/home/wang/video/data'+str(index)+'/lidar/'+pcd_ts+'.npy')
            input_img = get_img(img, nav)
            result = get_net_result(input_img)[0][0]
            result = result.data.cpu().numpy()*255+255
            # inverse_perspective_mapping(result, pcd)
            # result = cv2.resize(result, (img.shape[1], img.shape[0]))
            # mask = np.where((result > 200))
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

            # mask_img = np.zeros((img.shape), dtype=np.uint8)
            
            # mask_img[mask[0], mask[1]] = (255, 0, 0)
            # img = cv2.addWeighted(img, 1, mask_img, 0.8, 0)

            costmaps = []
            step = 1
            for i in range(0, step*8, step):
                costmap_ts = find_nn(ts, choose_dataset['cost_list'], i)
                # costmap = cv2.imread('/home/wang/video/data'+str(index)+'/cost/'+costmap_ts+'.png')
                costmap = Image.open('/home/wang/video/data'+str(index)+'/cost/'+costmap_ts+'.png')
                costmaps.append(costmap)

            v0 = 6.
            query_num = 32
            t = np.linspace(0.05, 0.6, query_num)
            v0 = torch.FloatTensor([v0]).expand(len(t),1)
            t = torch.FloatTensor(t).view(len(t),1)
            t.requires_grad = True
            
            costmaps = [costmap_transforms(costmap) for costmap in costmaps]
            costmaps = torch.stack(tuple(costmaps), dim=0).expand(len(t),8,1,200, 400)
            costmaps.requires_grad = True
            output = model(costmaps.to(device), t.to(device), v0.to(device))
            x = output[:,0] *30.
            y = output[:,1] *60.
            # vx = grad(output[:,0].sum(), t, create_graph=True)[0] *(30./3)
            # vy = grad(output[:,1].sum(), t, create_graph=True)[0] *(30./3)
            # ax = grad(vx.sum(), t, create_graph=True)[0] *(1./3)
            # ay = grad(vy.sum(), t, create_graph=True)[0] *(1./3)
            x = x.data.cpu().numpy()
            y = y.data.cpu().numpy()
            # vx = vx.data.cpu().numpy()
            # vy = vy.data.cpu().numpy()
            # ax = ax.data.cpu().numpy()
            # ay = ay.data.cpu().numpy()
            output_shape = result.shape

            z = np.ones(query_num)*-1.8
            pd = np.stack([x, y, z])

            img2 = lidar2camera(pd)
            show_img = cv2.addWeighted(img, 1, img2, 0.5, 0)
            cv2.imwrite('/home/wang/video/data'+str(index)+'/output/'+ts+'.png', show_img)
            cv2.imshow('result', show_img)
            cv2.waitKey(1)

            costmap = []