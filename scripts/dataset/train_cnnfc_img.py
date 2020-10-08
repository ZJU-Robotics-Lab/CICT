#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from os.path import join, dirname
sys.path.insert(0, join(dirname(__file__), '..'))
sys.path.insert(0, join(dirname(__file__), '../../'))

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
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from learning.path_model import CNNFC
from utils import write_params, fig2data

import carla_utils as cu
from learning.robo_dataset_utils.robo_utils.kitti.torch_dataset import TrajectoryDatasetImage

global_trajectory = None
global_trajectory_real = None

random.seed(datetime.now())
torch.manual_seed(233)
torch.cuda.manual_seed(233)
torch.set_num_threads(16)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--test_mode', type=bool, default=False, help='test model switch')
parser.add_argument('--dataset_name', type=str, default="kitti-test-image-01", help='name of the dataset')
parser.add_argument('--width', type=int, default=400, help='image width')
parser.add_argument('--height', type=int, default=200, help='image height')
parser.add_argument('--scale', type=float, default=30., help='longitudinal length')
parser.add_argument('--batch_size', type=int, default=24, help='size of the batches')
parser.add_argument('--traj_steps', type=int, default=8, help='traj steps')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='adam: weight_decay')
parser.add_argument('--lr', type=float, default=3e-4, help='adam: learning rate')
parser.add_argument('--gamma', type=float, default=0.01, help='xy and vxy loss trade off')
parser.add_argument('--img_step', type=int, default=3, help='RNN input image step')
parser.add_argument('--n_cpu', type=int, default=16, help='number of cpu threads to use during batch generation')
parser.add_argument('--checkpoint_interval', type=int, default=2000, help='interval between model checkpoints')
parser.add_argument('--test_interval', type=int, default=50, help='interval between model test')
parser.add_argument('--max_dist', type=float, default=25., help='max distance')
parser.add_argument('--max_speed', type=float, default=15., help='max distance')
parser.add_argument('--max_t', type=float, default=3., help='max time')
opt = parser.parse_args()
if opt.test_mode: opt.batch_size = 1
    
description = 'change costmap'
log_path = 'result/log/'+opt.dataset_name+'/'
os.makedirs('result/saved_models/%s' % opt.dataset_name, exist_ok=True)
os.makedirs('result/output/%s' % opt.dataset_name, exist_ok=True)
if not opt.test_mode:
    logger = SummaryWriter(log_dir=log_path)
    write_params(log_path, parser, description)
    
model = CNNFC(hidden_dim=256, input_dim=3).to(device)
model.load_state_dict(torch.load('result/saved_models/kitti-train-image-01/model_4000.pth'))
criterion = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

param = cu.parse_yaml_file_unsafe('../../learning/robo_dataset_utils/params/param_kitti.yaml')
trajectory_dataset = TrajectoryDatasetImage(param, 'train')#7
dataloader = DataLoader(trajectory_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)

eval_trajectory_dataset = TrajectoryDatasetImage(param, 'eval')#2
dataloader_eval = DataLoader(eval_trajectory_dataset, batch_size=1, shuffle=False, num_workers=1)
eval_samples = iter(dataloader_eval)


def show_traj(step):
    global global_trajectory, global_trajectory_real
    max_x = 30.
    max_y = 30.
    max_speed = 20.0

    trajectory = global_trajectory
    real_trajectory = global_trajectory_real
    fig = plt.figure(figsize=(7, 7))
    ax1 = fig.add_subplot(111)
    x = trajectory['x']
    y = trajectory['y']
    real_x = real_trajectory['x']
    real_y = real_trajectory['y']
    ax1.plot(x, y, label='trajectory', color = 'b', linewidth=5)
    ax1.plot(real_x, real_y, label='real-trajectory', color = 'b', linewidth=5, linestyle='--')
    ax1.set_xlabel('Forward/(m)')
    ax1.set_ylabel('Sideways/(m)')  
    ax1.set_xlim([0., max_x])
    ax1.set_ylim([-max_y, max_y])
    plt.legend(loc='lower right')
    
    #t = max_x*np.arange(0, 1.0, 1./x.shape[0])
    
    real_t = max_x*global_trajectory_real['ts_list']
    vx = trajectory['vx']
    vy = trajectory['vy']
    real_vx = real_trajectory['vx']
    real_vy = real_trajectory['vy']
    #v = np.sqrt(np.power(vx, 2), np.power(vy, 2))
    #angle = np.arctan2(vy, vx)/np.pi*max_speed
    #real_v = np.sqrt(np.power(real_vx, 2), np.power(real_vy, 2))

    #real_angle = np.arctan2(real_vy, real_vx)/np.pi*max_speed
    ax2 = ax1.twinx()
    ax2.plot(real_t, vx, label='vx', color = 'tab:cyan', linewidth=2)
    ax2.plot(real_t, vy, label='vy', color = 'tab:purple', linewidth=2)
    ax2.plot(real_t, real_vx, label='real-vx', color = 'r', linewidth=2, linestyle='--')
    ax2.plot(real_t, real_vy, label='real-vy', color = 'g', linewidth=2, linestyle='--')
    
    #ax2.plot(real_t, v, label='speed', color = 'r', linewidth=2)
    #ax2.plot(real_t, angle, label='angle', color = 'g', linewidth=2)
    # real
    #ax2.plot(real_t, real_v, label='real-speed', color = 'r', linewidth=2, linestyle='--')
    #ax2.plot(real_t, real_angle, label='real-angle', color = 'g', linewidth=2, linestyle='--')
    
    ax2.set_ylabel('Velocity/(m/s)')
    ax2.set_ylim([-max_speed, max_speed])
    plt.legend(loc='lower left')
    plt.close('all')
    
    img = fig2data(fig)
    cv2.imwrite('result/output/%s/' % opt.dataset_name+str(step)+'_curve.png', img)
    
def draw_traj(step):
    global global_trajectory, global_trajectory_real
    
    model.eval()
    batch = next(eval_samples)

    batch['img'] = batch['img'].to(device)
    batch['xy'] = batch['xy'].to(device)
    batch['vxy'] = batch['vxy'].to(device)
    
    ts_list = batch['t'].data.cpu().numpy()[0][:,0]
    
    output = model(batch['img'])
    output = output.view(1, 10, 4)
    x = output[:,:,0]*opt.max_dist
    y = output[:,:,1]*opt.max_dist
    vx = output[:,:,2]*opt.max_speed
    vy = output[:,:,3]*opt.max_speed

    real_x = batch['xy'].data.cpu().numpy()[0][:,0]*opt.max_dist
    real_y = batch['xy'].data.cpu().numpy()[0][:,1]*opt.max_dist
    real_vx = batch['vxy'].data.cpu().numpy()[0][:,0]
    real_vy = batch['vxy'].data.cpu().numpy()[0][:,1]

    x = x.data.cpu().numpy()[0]
    y = y.data.cpu().numpy()[0]
    vx = vx.data.cpu().numpy()[0]
    vy = vy.data.cpu().numpy()[0]
    
    global_trajectory = {'x':x, 'y':y, 'vx':vx, 'vy':vy}
    global_trajectory_real = {'x':real_x, 'y':real_y, 'vx':real_vx, 'vy':real_vy, 'ts_list':ts_list}
    show_traj(step)

    model.train()

def eval_error(total_step):
    model.eval()
    abs_x = []
    abs_y = []
    abs_vx = []
    abs_vy = []
    abs_v = []
    ade = []
    final_displacement = []
    for i in range(20):
        batch = next(eval_samples)
        batch['img'] = batch['img'].to(device)
        batch['xy'] = batch['xy'].to(device)
        batch['vxy'] = batch['vxy'].to(device)
    
        output = model(batch['img'])
        output = output.view(1, 10, 4)
        x = output[:,:,0]*opt.max_dist
        y = output[:,:,1]*opt.max_dist
        vx = output[:,:,2]*opt.max_speed
        vy = output[:,:,3]*opt.max_speed
    
        real_x = batch['xy'].data.cpu().numpy()[0][:,0]*opt.max_dist
        real_y = batch['xy'].data.cpu().numpy()[0][:,1]*opt.max_dist
        real_vx = batch['vxy'].data.cpu().numpy()[0][:,0]
        real_vy = batch['vxy'].data.cpu().numpy()[0][:,1]

        x = x.data.cpu().numpy()[0]
        y = y.data.cpu().numpy()[0]
        vx = vx.data.cpu().numpy()[0]
        vy = vy.data.cpu().numpy()[0]
        #print(x,real_x)
        abs_x.append(np.mean(np.abs(x-real_x)))
        abs_y.append(np.mean(np.abs(y-real_y)))
        abs_vx.append(np.mean(np.abs(vx - real_vx)))
        abs_vy.append(np.mean(np.abs(vy - real_vy)))
        final_displacement.append(np.abs(x-real_x)[-1])
        abs_v.append(np.mean(np.hypot(vx - real_vx, vy - real_vy)))
        ade.append(np.mean(np.hypot(x - real_x, y - real_y)))

    logger.add_scalar('eval/x', sum(abs_x)/len(abs_x), total_step)
    logger.add_scalar('eval/y', sum(abs_y)/len(abs_y), total_step)
    logger.add_scalar('eval/vx', sum(abs_vx)/len(abs_vx), total_step)
    logger.add_scalar('eval/vy', sum(abs_vy)/len(abs_vy), total_step)
    logger.add_scalar('eval/v', sum(abs_v)/len(abs_v), total_step)
    logger.add_scalar('eval/ade', sum(ade)/len(ade), total_step)
    logger.add_scalar('eval/final_displacement', sum(final_displacement)/len(final_displacement), total_step)
    model.train()
    
total_step = 0
print('Start to train ...')

for index, batch in enumerate(dataloader):
    total_step += 1
    if opt.test_mode:
        for j in range(500): draw_traj(j)
        break

    batch['img'] = batch['img'].to(device)
    batch['xy'] = batch['xy'].to(device)
    batch['vxy'] = batch['vxy'].to(device)/opt.max_speed
    
    output = model(batch['img'])
    output = output.view(opt.batch_size, 10, 4)
    xy = output[:,:,0:2]
    vxy = output[:,:,2:4]

    optimizer.zero_grad()
    loss_xy = criterion(opt.max_dist*xy, opt.max_dist*batch['xy'])
    #loss_x = criterion(opt.max_dist*xy[:,0], opt.max_dist*batch['xy'][:,0])
    #loss_y = criterion(opt.max_dist*xy[:,1], opt.max_dist*batch['xy'][:,1])
    loss_vxy = criterion(vxy, batch['vxy'])

    loss = loss_xy + opt.gamma*loss_vxy
    #loss = loss_x + 0.2*loss_y + opt.gamma*loss_vxy

    loss.backward()
    torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1)
    optimizer.step()

    logger.add_scalar('train/loss_xy', loss_xy.item(), total_step)
    #logger.add_scalar('train/loss_x', loss_x.item(), total_step)
    #logger.add_scalar('train/loss_y', loss_y.item(), total_step)
    logger.add_scalar('train/loss_vxy', loss_vxy.item(), total_step)
    logger.add_scalar('train/loss', loss.item(), total_step)
    
    if total_step % opt.test_interval == 0:
        eval_error(total_step)
        #draw_traj(total_step)
    
    if total_step % opt.checkpoint_interval == 0:
        torch.save(model.state_dict(), 'result/saved_models/%s/model_%d.pth'%(opt.dataset_name, total_step))