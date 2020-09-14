#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from os.path import join, dirname
sys.path.insert(0, join(dirname(__file__), '../../'))
sys.path.insert(0, join(dirname(__file__), '..'))

import os
import time
import random
import argparse
import numpy as np
from PIL import Image, ImageDraw
from datetime import datetime
import cv2
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})

import torch
from torch.autograd import grad
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from learning.path_model import Model_COS, ModelGRU
from learning.costmap_dataset import CostMapDataset
from utils import write_params, fig2data

random.seed(datetime.now())
torch.manual_seed(666)
torch.cuda.manual_seed(666)
torch.set_num_threads(16)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--test_mode', type=bool, default=True, help='test model switch')
parser.add_argument('--dataset_name', type=str, default="test-gru-01", help='name of the dataset')
parser.add_argument('--width', type=int, default=400, help='image width')
parser.add_argument('--height', type=int, default=200, help='image height')
parser.add_argument('--scale', type=float, default=25., help='longitudinal length')
parser.add_argument('--batch_size', type=int, default=32, help='size of the batches')
parser.add_argument('--img_step', type=int, default=3, help='RNN input image step')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='adam: weight_decay')
parser.add_argument('--lr', type=float, default=3e-4, help='adam: learning rate')
parser.add_argument('--n_cpu', type=int, default=16, help='number of cpu threads to use during batch generation')
parser.add_argument('--max_dist', type=float, default=25., help='max distance')
parser.add_argument('--max_t', type=float, default=3., help='max time')
opt = parser.parse_args()
if opt.test_mode: opt.batch_size = 1

    
description = 'gru-01'
os.makedirs('result/saved_models/%s' % opt.dataset_name, exist_ok=True)
os.makedirs('result/output/%s' % opt.dataset_name, exist_ok=True)


global_trajectory = None
global_trajectory_real = None
    
model = ModelGRU().to(device)
model.load_state_dict(torch.load('result/saved_models/mu-log_var-05/model_278000.pth'))
model.eval()

eval_loader = DataLoader(CostMapDataset(data_index=[1,2,3,4,5,6,7], opt=opt, dataset_path='/media/wang/DATASET/CARLA_HUMAN/town01/', evalmode=True), batch_size=1, shuffle=False, num_workers=1)
eval_samples = iter(eval_loader)


def xy2uv(x, y):
    pixs_per_meter = opt.height/opt.scale
    u = (opt.height-x*pixs_per_meter).astype(int)
    v = (y*pixs_per_meter+opt.width//2).astype(int)
    mask = np.where((u >= 0)&(u < opt.height))[0]
    u = u[mask]
    v = v[mask]
    mask = np.where((v >= 0)&(v < opt.width))[0]
    u = u[mask]
    v = v[mask]
    return u, v

def show_traj(step):
    global global_trajectory, global_trajectory_real
    max_x = 30.
    max_y = 30.
    max_speed = 12.0

    trajectory = global_trajectory
    real_trajectory = global_trajectory_real
    fig = plt.figure(figsize=(7, 7))
    ax1 = fig.add_subplot(111)
    x = trajectory['x']
    y = trajectory['y']
    real_x = real_trajectory['x']
    real_y = real_trajectory['y']
    #ax1.plot(x, y, label='trajectory', color = 'b', linewidth=5)
    ax1.plot(real_x, real_y, label='real-trajectory', color = 'b', linewidth=5, linestyle='--')
    ax1.set_xlabel('Forward/(m)')
    ax1.set_ylabel('Sideways/(m)')  
    ax1.set_xlim([0., max_x])
    ax1.set_ylim([-max_y, max_y])
    plt.legend(loc='lower right')
    
    t = max_speed*np.arange(0, 1.0, 1./x.shape[0])
    real_t = max_speed*global_trajectory_real['ts_list']
    a = trajectory['a']
    vx = trajectory['vx']
    vy = trajectory['vy']
    real_a = real_trajectory['a_list']
    real_vx = real_trajectory['vx']
    real_vy = real_trajectory['vy']
    v = np.sqrt(np.power(vx, 2), np.power(vy, 2))
    angle = np.arctan2(vy, vx)/np.pi*max_speed
    real_v = np.sqrt(np.power(real_vx, 2), np.power(real_vy, 2))
    real_angle = np.arctan2(real_vy, real_vx)/np.pi*max_speed
    ax2 = ax1.twinx()
    #ax2.plot(t, v, label='speed', color = 'r', linewidth=2)
    #ax2.plot(t, a, label='acc', color = 'y', linewidth=2)
    #ax2.plot(t, angle, label='angle', color = 'g', linewidth=2)
    # real
    ax2.plot(real_t, real_v, label='real-speed', color = 'r', linewidth=2, linestyle='--')
    ax2.plot(real_t, real_a, label='real-acc', color = 'y', linewidth=2, linestyle='--')
    ax2.plot(real_t, real_angle, label='real-angle', color = 'g', linewidth=2, linestyle='--')
    
    ax2.set_ylabel('Velocity/(m/s)')
    ax2.set_ylim([-max_speed, max_speed])
    plt.legend(loc='lower left')
    #plt.show()
    plt.close('all')
    
    img = fig2data(fig)
    cv2.imwrite('result/output/%s/' % opt.dataset_name+str(step)+'_curve.png', img)

def draw_traj(step):
    if opt.test_mode: global global_trajectory, global_trajectory_real
    
    model.eval()
    batch = next(eval_samples)
    img = batch['img'][:,-1,:].clone().data.numpy().squeeze()*127+128

    t = torch.arange(0, 0.99, 0.05).unsqueeze(1).to(device)
    t.requires_grad = True

    batch['img'] = batch['img'].expand(len(t),10,1,opt.height, opt.width)
    batch['img'] = batch['img'].to(device)
    batch['t'] = batch['t'].to(device)
    batch['v_0'] = batch['v_0'].expand(len(t),1)
    batch['v_0'] = batch['v_0'].to(device)
    batch['xy'] = batch['xy'].to(device)
    batch['vxy'] = batch['vxy'].to(device)
    batch['img'].requires_grad = True
    batch['t'].requires_grad = True
    
    output = model(batch['img'], t, batch['v_0'])
    vx = grad(output[:,0].sum(), t, create_graph=True)[0][:,0]*(opt.max_dist/opt.max_t)
    vy = grad(output[:,1].sum(), t, create_graph=True)[0][:,0]*(opt.max_dist/opt.max_t)
    
    ax = grad(vx.sum(), t, create_graph=True)[0][:,0]/opt.max_t
    ay = grad(vy.sum(), t, create_graph=True)[0][:,0]/opt.max_t

    output_axy = torch.cat([ax.unsqueeze(1), ay.unsqueeze(1)], dim=1)

    x = output[:,0]*opt.max_dist
    y = output[:,1]*opt.max_dist
    
    theta_a = torch.atan2(ay, ax)
    theta_v = torch.atan2(vy, vx)
    sign = torch.sign(torch.cos(theta_a-theta_v))
    a = torch.mul(torch.norm(output_axy, dim=1), sign.flatten()).unsqueeze(1)
    
    vx = vx.data.cpu().numpy()
    vy = vy.data.cpu().numpy()
    x = x.data.cpu().numpy()
    y = y.data.cpu().numpy()
    ax = ax.data.cpu().numpy()
    ay = ay.data.cpu().numpy()
    a = a.data.cpu().numpy()
    
    real_x = batch['x_list'].data.cpu().numpy()[0]
    real_y = batch['y_list'].data.cpu().numpy()[0]
    real_vx = batch['vx_list'].data.cpu().numpy()[0]
    real_vy = batch['vy_list'].data.cpu().numpy()[0]
    ts_list = batch['ts_list'].data.cpu().numpy()[0]
    a_list = batch['a_list'].data.cpu().numpy()[0]

    global_trajectory = {'x':x, 'y':y, 'vx':vx, 'vy':vy, 'a':a}
    global_trajectory_real = {'x':real_x, 'y':real_y, 'vx':real_vx, 'vy':real_vy, 'ts_list':ts_list, 'a_list':a_list}
    show_traj(step)
        
    img = Image.fromarray(img).convert("RGB")
    draw =ImageDraw.Draw(img)
    
    real_x = batch['x_list'].squeeze().data.numpy()
    real_y = batch['y_list'].squeeze().data.numpy()
    real_u, real_v = xy2uv(real_x, real_y)
    
    for i in range(len(real_u)-1):
        draw.line((real_v[i], real_u[i], real_v[i+1], real_u[i+1]), 'blue')
        draw.line((real_v[i]+1, real_u[i], real_v[i+1]+1, real_u[i+1]), 'blue')
        draw.line((real_v[i]-1, real_u[i], real_v[i+1]-1, real_u[i+1]), 'blue')
        
    result = output.data.cpu().numpy()
    x = opt.max_dist*result[:,0]
    y = opt.max_dist*result[:,1]
    u, v = xy2uv(x, y)

    for i in range(len(u)-1):
        draw.line((v[i], u[i], v[i+1], u[i+1]), 'red')
        draw.line((v[i]+1, u[i], v[i+1]+1, u[i+1]), 'red')
        draw.line((v[i]-1, u[i], v[i+1]-1, u[i+1]), 'red')
    
    img.save(('result/output/%s/' % opt.dataset_name)+str(step)+'_costmap.png')
    model.train()

for j in range(1000):
    draw_traj(j)