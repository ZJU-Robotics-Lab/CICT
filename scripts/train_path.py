#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from os.path import join, dirname
sys.path.insert(0, join(dirname(__file__), '..'))

import os
import glob
import random
import argparse
import numpy as np
from PIL import Image
from datetime import datetime

import torch
from torch.autograd import grad
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from learning.path_model import Model

random.seed(datetime.now())
torch.manual_seed(999)
torch.cuda.manual_seed(999)
torch.set_num_threads(16)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default="test_no_vel_02", help='name of the dataset')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--lr', type=float, default=3e-4, help='adam: learning rate')
parser.add_argument('--gamma', type=float, default=0.1, help='xy and vxy trade off')
parser.add_argument('--n_cpu', type=int, default=16, help='number of cpu threads to use during batch generation')
parser.add_argument('--checkpoint_interval', type=int, default=1000, help='interval between model checkpoints')
parser.add_argument('--test_interval', type=int, default=10, help='interval between model test')
opt = parser.parse_args()

log_path = 'log/'+opt.dataset_name+'/'
os.makedirs('result/saved_models/%s' % opt.dataset_name, exist_ok=True)
logger = SummaryWriter(log_dir=log_path)

class CostMapDataset(Dataset):
    def __init__(self, data_index=[1,4,5,8]):
        self.data_index = data_index
        self.max_dist = 20
        self.max_t = 5.0
        transforms_ = [ transforms.Resize((200, 400), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
            ]
        
        self.transform = transforms.Compose(transforms_)
        self.dataset_path = '/media/wang/DATASET/CARLA/'
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
            file_name = file.split('/')[7][:-4]
            file_names.append(file_name)
        file_names.sort()
        self.files_dict[index] = file_names

    def __getitem__(self, index):
        data_index = random.sample(self.data_index, 1)[0]
        flag = True
        while flag:
            file_name = random.sample(self.files_dict[data_index][500:-1000], 1)[0]
            image_path = '/media/wang/DATASET/CARLA/'+str(data_index)+'/ipm/'+file_name+'.png'
            img = Image.open(image_path).convert('L')
            img = self.transform(img)
            
            x_0 = self.pose_dict[data_index][file_name][0]
            y_0 = self.pose_dict[data_index][file_name][1]
            yaw = np.deg2rad(self.pose_dict[data_index][file_name][3])
            
            self.files_dict[data_index].sort()
            ts_index = self.files_dict[data_index].index(file_name)
            ts_list = []

            for i in range(ts_index+1, len(self.files_dict[data_index])-500):
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
                        ts_list.append(ts)
            if len(ts_list) == 0:
                continue
            else:
                ts = random.sample(ts_list, 1)[0]
                break
        
        t = torch.FloatTensor([float(ts) - float(file_name)])

        x_t = self.pose_dict[data_index][ts][0]
        y_t = self.pose_dict[data_index][ts][1]

        dx = x_t - x_0
        dy = y_t - y_0
        
        x = np.cos(yaw)*dx + np.sin(yaw)*dy
        y = np.cos(yaw)*dy - np.sin(yaw)*dx
        # [-1, 1]
        xy = torch.FloatTensor([x/self.max_dist, y/self.max_dist])
        
        _vx = self.vel_dict[data_index][ts][0]
        _vy = self.vel_dict[data_index][ts][1]
        vx = np.cos(yaw)*_vx + np.sin(yaw)*_vy
        vy = np.cos(yaw)*_vy - np.sin(yaw)*_vx
        
        vxy = torch.FloatTensor([vx/self.max_dist, vy/self.max_dist])

        return {'img': img, 't': t, 'xy':xy, 'vxy':vxy}

    def __len__(self):
        return 100000000000
    
def test_model(total_step):
    model.eval()
    
    batch = next(test_samples)
    batch['img'] = batch['img'].to(device)
    batch['t'] = batch['t'].to(device)
    batch['xy'] = batch['xy'].to(device)
    batch['vxy'] = batch['vxy'].to(device)
    batch['img'].requires_grad = True
    batch['t'].requires_grad = True

    output = model(batch['img'], batch['t'])
    vx = grad(output[:,0].sum(), batch['t'], create_graph=True)[0]
    vy = grad(output[:,1].sum(), batch['t'], create_graph=True)[0]
    output_vxy = torch.cat([vx, vy], dim=1)
    
    loss_xy = criterion(output, batch['xy'])
    loss_vxy = criterion(output_vxy, batch['vxy'])
    loss = loss_xy + opt.gamma*loss_vxy

    logger.add_scalar('test/loss_xy', loss_xy.item(), total_step)
    logger.add_scalar('test/loss_vxy', loss_vxy.item(), total_step)
    logger.add_scalar('test/loss', loss.item(), total_step)
    model.train()

model = Model().to(device)
#model.load_state_dict(torch.load('result/saved_models/test_no_vel_01/model_10000.pth'))
train_loader = DataLoader(CostMapDataset(), batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)
test_loader = DataLoader(CostMapDataset(data_index=[10]), batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)
test_samples = iter(test_loader)

criterion = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

total_step = 0
#abs_x = []
#abs_y = []
print('Start !')
for i, batch in enumerate(train_loader):
    batch['img'] = batch['img'].to(device)
    batch['t'] = batch['t'].to(device)
    batch['xy'] = batch['xy'].to(device)
    batch['vxy'] = batch['vxy'].to(device)
    batch['img'].requires_grad = True
    batch['t'].requires_grad = True

    output = model(batch['img'], batch['t'])
    vx = grad(output[:,0].sum(), batch['t'], create_graph=True)[0]
    vy = grad(output[:,1].sum(), batch['t'], create_graph=True)[0]
    output_vxy = torch.cat([vx, vy], dim=1)

    optimizer.zero_grad()
    loss_xy = criterion(output, batch['xy'])
    loss_vxy = criterion(output_vxy, batch['vxy'])
    loss = loss_xy + opt.gamma*loss_vxy
    """
    gen = output.data.cpu().numpy()[0]
    real = batch['xy'].data.cpu().numpy()[0]
    abs_x.append(abs(gen[0]-real[0]))
    abs_y.append(abs(gen[1]-real[1]))
    """
    loss.backward()
    torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1)
    optimizer.step()
    
    logger.add_scalar('train/loss_xy', loss_xy.item(), total_step)
    logger.add_scalar('train/loss_vxy', loss_vxy.item(), total_step)
    logger.add_scalar('train/loss', loss.item(), total_step)
    
    if total_step % opt.test_interval == 0:
        test_model(total_step)
        
    if total_step % opt.checkpoint_interval == 0:
        torch.save(model.state_dict(), 'result/saved_models/%s/model_%d.pth'%(opt.dataset_name, total_step))
    total_step += 1
    #if total_step == 500:
    #    break
#print(sum(abs_x)/len(abs_x), sum(abs_y)/len(abs_y))