#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from os.path import join, dirname
sys.path.insert(0, join(dirname(__file__), '..'))

import os
import glob
import time
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

from learning.path_model import Model, VAE, CNN_SIN

random.seed(datetime.now())
torch.manual_seed(666)
torch.cuda.manual_seed(666)
torch.set_num_threads(16)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--test_mode', type=bool, default=False, help='test model switch')
parser.add_argument('--dataset_name', type=str, default="with_v0_01", help='name of the dataset')
parser.add_argument('--batch_size', type=int, default=128, help='size of the batches')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='adam: weight_decay')
parser.add_argument('--lr', type=float, default=3e-4, help='adam: learning rate')
parser.add_argument('--gamma', type=float, default=0.2, help='xy and vxy loss trade off')
parser.add_argument('--gamma2', type=float, default=0., help='KLD loss trade off')
parser.add_argument('--n_cpu', type=int, default=16, help='number of cpu threads to use during batch generation')
parser.add_argument('--checkpoint_interval', type=int, default=2000, help='interval between model checkpoints')
parser.add_argument('--test_interval', type=int, default=50, help='interval between model test')
parser.add_argument('--max_dist', type=float, default=20., help='max distance')
parser.add_argument('--max_t', type=float, default=5., help='max time')
opt = parser.parse_args()
if opt.test_mode: opt.batch_size = 1

log_path = 'log/'+opt.dataset_name+'/'
os.makedirs('result/saved_models/%s' % opt.dataset_name, exist_ok=True)

def write_params():
    with open(log_path+'params.cfg', 'w+') as file:
        file.write('********************************')
        file.write('time: '+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+'\n')
        file.write('dataset_name: ' + str(opt.dataset_name)+'\n')
        file.write('batch_size: ' + str(opt.batch_size)+'\n')
        file.write('lr: ' + str(opt.lr)+'\n')
        file.write('weight_decay: ' + str(opt.weight_decay)+'\n')
        file.write('vxy loss: ' + str(opt.gamma)+'\n')
        file.write('kld loss: ' + str(opt.gamma2)+'\n')
        file.write('max_dist: ' + str(opt.max_dist)+'\n')
        file.write('********************************\n\n')
        
class CostMapDataset(Dataset):
    def __init__(self, data_index=[1,4,5,8]):
        self.data_index = data_index
        self.max_dist = opt.max_dist
        self.max_t = opt.max_t
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
        # [0 ~ 1]
        t = torch.FloatTensor([float(ts)/self.max_t - float(file_name)/self.max_t])
        _vx_0 = self.vel_dict[data_index][file_name][0]
        _vy_0 = self.vel_dict[data_index][file_name][1]
        vx_0 = np.cos(yaw)*_vx_0 + np.sin(yaw)*_vy_0
        #vy_0 = np.cos(yaw)*_vy_0 - np.sin(yaw)*_vx_0
        v_0 = torch.FloatTensor([vx_0])
        
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
        
        vxy = torch.FloatTensor([vx, vy])

        return {'img': img, 't': t, 'xy':xy, 'vxy':vxy, 'v_0':v_0}

    def __len__(self):
        return 100000000000
    
def test_model(total_step):
    model.eval()
    
    batch = next(test_samples)
    batch['img'] = batch['img'].to(device)
    batch['t'] = batch['t'].to(device)
    batch['v_0'] = batch['v_0'].to(device)
    batch['xy'] = batch['xy'].to(device)
    batch['vxy'] = batch['vxy'].to(device)
    batch['img'].requires_grad = True
    batch['t'].requires_grad = True

    #output = model(batch['img'], batch['t'])
    output = model(batch['img'], batch['t'], batch['v_0'])
    vx = grad(output[:,0].sum(), batch['t'], create_graph=True)[0]
    vy = grad(output[:,1].sum(), batch['t'], create_graph=True)[0]
    output_vxy = (opt.max_dist/opt.max_t)*torch.cat([vx, vy], dim=1)
    
    loss_xy = criterion(output, batch['xy'])
    loss_vxy = criterion(output_vxy, batch['vxy'])
    #KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    loss = loss_xy + opt.gamma*loss_vxy# + opt.gamma2*KLD

    logger.add_scalar('test/loss_xy', opt.max_dist*loss_xy.item(), total_step)
    logger.add_scalar('test/loss_vxy', loss_vxy.item(), total_step)
    #logger.add_scalar('test/loss_kld', KLD.item(), total_step)
    logger.add_scalar('test/loss', loss.item(), total_step)
    model.train()

def eval_error(total_step):
    model.eval()
    abs_x = []
    abs_y = []
    abs_vx = []
    abs_vy = []
    rel_x = []
    rel_y = []
    rel_vx = []
    rel_vy = []
    for i in range(100):
        batch = next(eval_samples)
        batch['img'] = batch['img'].to(device)
        batch['t'] = batch['t'].to(device)
        batch['v_0'] = batch['v_0'].to(device)
        batch['xy'] = batch['xy'].to(device)
        batch['vxy'] = batch['vxy'].to(device)
        batch['img'].requires_grad = True
        batch['t'].requires_grad = True
    
        #output = model(batch['img'], batch['t'])
        output = model(batch['img'], batch['t'], batch['v_0'])
        vx = (opt.max_dist/opt.max_t)*grad(output[:,0].sum(), batch['t'], create_graph=True)[0]
        vy = (opt.max_dist/opt.max_t)*grad(output[:,1].sum(), batch['t'], create_graph=True)[0]
        
        gen = output.data.cpu().numpy()[0]
        real = batch['xy'].data.cpu().numpy()[0]
        vx = vx.data.cpu().numpy()[0][0]
        vy = vy.data.cpu().numpy()[0][0]
        real_v = batch['vxy'].data.cpu().numpy()[0]
        abs_x.append(abs(gen[0]-real[0]))
        abs_y.append(abs(gen[1]-real[1]))
        abs_vx.append(abs(vx - real_v[0]))
        abs_vy.append(abs(vy - real_v[1]))
        if abs(real[0]) > 0.05:
            rel_x.append(abs(gen[0]-real[0])/(abs(real[0])))
        if abs(real[1]) > 0.05:
            rel_y.append(abs(gen[1]-real[1])/(abs(real[1])))
        if abs(real_v[0]) > 0.05:
            rel_vx.append(abs(vx - real_v[0])/(abs(real_v[0])))
        if abs(real_v[1]) > 0.05:
            rel_vy.append(abs(vy - real_v[1])/(abs(real_v[1])))

    logger.add_scalar('eval/x', opt.max_dist*sum(abs_x)/len(abs_x), total_step)
    logger.add_scalar('eval/y', opt.max_dist*sum(abs_y)/len(abs_y), total_step)
    logger.add_scalar('eval/vx', sum(abs_vx)/len(abs_vx), total_step)
    logger.add_scalar('eval/vy', sum(abs_vy)/len(abs_vy), total_step)

    if len(rel_x) > 0:
        logger.add_scalar('eval/rel_x', sum(rel_x)/len(rel_x), total_step)
    if len(rel_y) > 0:
        logger.add_scalar('eval/rel_y', sum(rel_y)/len(rel_y), total_step)
    if len(rel_vx) > 0:
        logger.add_scalar('eval/rel_vx', sum(rel_vx)/len(rel_vx), total_step)
    if len(rel_vy) > 0:
        logger.add_scalar('eval/rel_vy', sum(rel_vy)/len(rel_vy), total_step)
    model.train()
    
model = Model().to(device)
#model.load_state_dict(torch.load('result/saved_models/VAE01/model_17000.pth'))
train_loader = DataLoader(CostMapDataset(), batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)
test_loader = DataLoader(CostMapDataset(data_index=[10]), batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)
test_samples = iter(test_loader)
eval_loader = DataLoader(CostMapDataset(data_index=[10]), batch_size=1, shuffle=False, num_workers=1)
eval_samples = iter(eval_loader)

criterion = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

if not opt.test_mode:
    logger = SummaryWriter(log_dir=log_path)
    write_params()

total_step = 0
if opt.test_mode:
    abs_x = []
    abs_y = []
    
print('Start to train ...')
for i, batch in enumerate(train_loader):
    total_step += 1
    if opt.test_mode: batch = next(test_samples)

    batch['img'] = batch['img'].to(device)
    batch['t'] = batch['t'].to(device)
    batch['v_0'] = batch['v_0'].to(device)
    batch['xy'] = batch['xy'].to(device)
    batch['vxy'] = batch['vxy'].to(device)
    batch['img'].requires_grad = True
    batch['t'].requires_grad = True

    output = model(batch['img'], batch['t'], batch['v_0'])
    vx = grad(output[:,0].sum(), batch['t'], create_graph=True)[0]
    vy = grad(output[:,1].sum(), batch['t'], create_graph=True)[0]
    output_vxy = (opt.max_dist/opt.max_t)*torch.cat([vx, vy], dim=1)

    optimizer.zero_grad()
    loss_xy = criterion(output, batch['xy'])
    loss_vxy = criterion(output_vxy, batch['vxy'])
    #KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    loss = loss_xy + opt.gamma*loss_vxy# + opt.gamma2*KLD
    
    if opt.test_mode:
        gen = output.data.cpu().numpy()[0]
        real = batch['xy'].data.cpu().numpy()[0]
        abs_x.append(abs(gen[0]-real[0]))
        abs_y.append(abs(gen[1]-real[1]))
        if total_step == 500:
            break
    else:
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1)
        optimizer.step()
    
        logger.add_scalar('train/loss_xy', opt.max_dist*loss_xy.item(), total_step)
        logger.add_scalar('train/loss_vxy', loss_vxy.item(), total_step)
        #logger.add_scalar('train/loss_kld', KLD.item(), total_step)
        logger.add_scalar('train/loss', loss.item(), total_step)
    
        if total_step % opt.test_interval == 0:
            test_model(total_step)
            eval_error(total_step)
        
        if total_step % opt.checkpoint_interval == 0:
            torch.save(model.state_dict(), 'result/saved_models/%s/model_%d.pth'%(opt.dataset_name, total_step))
        
if opt.test_mode: print('x:', opt.max_dist*sum(abs_x)/len(abs_x), 'm, y:', opt.max_dist*sum(abs_y)/len(abs_y), 'm')