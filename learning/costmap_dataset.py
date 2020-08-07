#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import glob
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

def angle_normal(angle):
    while angle >= np.pi:
        angle -= 2*np.pi
    while angle <= -np.pi:
        angle += 2*np.pi
    return angle

class CostMapDataset(Dataset):
    def __init__(self, data_index, opt, dataset_path='/media/wang/DATASET/CARLA/town01/', evalmode=False):
        self.evalmode = evalmode
        self.data_index = data_index
        self.max_dist = opt.max_dist
        self.max_t = opt.max_t
        transforms_ = [ transforms.Resize((200, 400), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
            ]
        
        self.transform = transforms.Compose(transforms_)
        self.dataset_path = dataset_path
        self.pose_dict = {}
        self.vel_dict = {}
        self.files_dict = {}
        self.total_len = 0
        
        for index in self.data_index:
            self.read_pose(index)
            self.read_vel(index)
            self.read_img(index)
        
    def read_pose(self, index):
        file_path = self.dataset_path+str(index)+'/state/pos.txt'
        ts_dict = {}
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                sp_line = line.split()
                ts = sp_line[0]
                x = float(sp_line[1])
                y = float(sp_line[2])
                z = float(sp_line[3])
                yaw = float(sp_line[5])
                ts_dict[ts] = [x, y, z, yaw]
        self.pose_dict[index] = ts_dict
        
    def read_vel(self, index):
        file_path = self.dataset_path+str(index)+'/state/vel.txt'
        ts_dict = {}
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                sp_line = line.split()
                ts = sp_line[0]
                vx = float(sp_line[1])
                vy = float(sp_line[2])
                vz = float(sp_line[3])
                ts_dict[ts] = [vx, vy, vz]
        self.vel_dict[index] = ts_dict
    
    def read_img(self, index):
        files = glob.glob(self.dataset_path+str(index)+'/ipm/*.png')
        file_names = []
        for file in files:
            file_name = file.split('/')[8][:-4]
            file_names.append(file_name)
        file_names.sort()
        self.files_dict[index] = file_names

    def tf_pose(self, data_index, ts, yaw, x_0, y_0):
        x_t = self.pose_dict[data_index][ts][0]
        y_t = self.pose_dict[data_index][ts][1]
        dx = x_t - x_0
        dy = y_t - y_0
        x = np.cos(yaw)*dx + np.sin(yaw)*dy
        y = np.cos(yaw)*dy - np.sin(yaw)*dx
        return x, y
        
    def __getitem__(self, index):
        data_index = random.sample(self.data_index, 1)[0]
        while True:
            file_name = random.sample(self.files_dict[data_index][:-120], 1)[0]
            image_path = self.dataset_path + str(data_index)+'/ipm/'+file_name+'.png'
            img = Image.open(image_path).convert('L')
            img = self.transform(img)
            
            x_0 = self.pose_dict[data_index][file_name][0]
            y_0 = self.pose_dict[data_index][file_name][1]
            yaw = np.deg2rad(self.pose_dict[data_index][file_name][3])
            
            self.files_dict[data_index].sort()
            ts_index = self.files_dict[data_index].index(file_name)
            ts_list = []
            x_list = []
            y_list = []
            for i in range(ts_index+1, len(self.files_dict[data_index])-100):
                ts = self.files_dict[data_index][i]
                
                _x_t = self.pose_dict[data_index][ts][0]
                _y_t = self.pose_dict[data_index][ts][1]
                distance = np.sqrt((x_0-_x_t)**2+(y_0-_y_t)**2)
                if distance > self.max_dist or (float(ts)-float(file_name) > self.max_t):
                    break
                else:
                    if distance < 0.03:
                        pass
                    else:
                        x_, y_ = self.tf_pose(data_index, ts, yaw, x_0, y_0)
                        x_list.append(x_)
                        y_list.append(y_)
                        ts_list.append(ts)
                        
            if len(ts_list) == 0:
                continue
            else:
                ts = random.sample(ts_list, 1)[0]
                break
        # [0 ~ 1]
        t = torch.FloatTensor([float(ts)/self.max_t - float(file_name)/self.max_t])
        _vx_0 = self.vel_dict[data_index][file_name][0]
        _vy_0 = self.vel_dict[data_index][file_name][1]
        vx_0 = np.cos(yaw)*_vx_0 + np.sin(yaw)*_vy_0
        #vy_0 = np.cos(yaw)*_vy_0 - np.sin(yaw)*_vx_0
        v_0 = torch.FloatTensor([vx_0])
        x, y = self.tf_pose(data_index, ts, yaw, x_0, y_0)
        # [-1, 1]
        xy = torch.FloatTensor([x/self.max_dist, y/self.max_dist])
        
        yaw_t = angle_normal(np.deg2rad(self.pose_dict[data_index][ts][3]) - yaw)
        # [-1, 1]
        yaw_t = torch.FloatTensor([yaw_t/np.pi])
    
        _vx = self.vel_dict[data_index][ts][0]
        _vy = self.vel_dict[data_index][ts][1]
        vx = np.cos(yaw)*_vx + np.sin(yaw)*_vy
        vy = np.cos(yaw)*_vy - np.sin(yaw)*_vx
        
        vxy = torch.FloatTensor([vx, vy])
        x_list = torch.FloatTensor(x_list)
        y_list = torch.FloatTensor(y_list)
        if self.evalmode:
            return {'img': img, 't': t, 'xy':xy, 'vxy':vxy, 'v_0':v_0, 'yaw_t': yaw_t, 'x_list':x_list, 'y_list':y_list}
        else:
            return {'img': img, 't': t, 'xy':xy, 'vxy':vxy, 'v_0':v_0}

    def __len__(self):
        return 100000000000