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
from torch.autograd import grad
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from learning.path_model import CIL
from utils import write_params, fig2data

import carla_utils as cu
from learning.robo_dataset_utils.robo_utils.kitti.torch_dataset import TrajectoryDataset3

global_trajectory = None
global_trajectory_real = None

random.seed(datetime.now())
torch.manual_seed(233)
torch.cuda.manual_seed(233)
torch.set_num_threads(16)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--test_mode', type=bool, default=False, help='test model switch')
parser.add_argument('--dataset_name', type=str, default="kitti-train-CIL-01", help='name of the dataset')
parser.add_argument('--width', type=int, default=400, help='image width')
parser.add_argument('--height', type=int, default=200, help='image height')
parser.add_argument('--scale', type=float, default=30., help='longitudinal length')
parser.add_argument('--batch_size', type=int, default=4, help='size of the batches')
parser.add_argument('--traj_steps', type=int, default=8, help='traj steps')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='adam: weight_decay')
parser.add_argument('--lr', type=float, default=3e-4, help='adam: learning rate')
parser.add_argument('--gamma', type=float, default=0.01, help='xy and vxy loss trade off')
parser.add_argument('--gamma2', type=float, default=0.01, help='xy and axy loss trade off')
parser.add_argument('--img_step', type=int, default=3, help='RNN input image step')
parser.add_argument('--n_cpu', type=int, default=4, help='number of cpu threads to use during batch generation')
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
    
model = CIL(256, 3).to(device)
#model.load_state_dict(torch.load('result/saved_models/kitti-train-cnnfc-02/model_34000.pth'))
criterion = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

param = cu.parse_yaml_file_unsafe('../../learning/robo_dataset_utils/params/param_kitti.yaml')


s_trajectory_dataset = TrajectoryDataset3(param, 'train', 's')#7
s_dataloader = DataLoader(s_trajectory_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)
s_train_samples = iter(s_dataloader)
l_trajectory_dataset = TrajectoryDataset3(param, 'train', 'l')#7
l_dataloader = DataLoader(l_trajectory_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)
l_train_samples = iter(l_dataloader)
r_trajectory_dataset = TrajectoryDataset3(param, 'train', 'r')#7
r_dataloader = DataLoader(r_trajectory_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)
r_train_samples = iter(r_dataloader)

s_eval_trajectory_dataset = TrajectoryDataset3(param, 'eval','s')#2
s_dataloader_eval = DataLoader(s_eval_trajectory_dataset, batch_size=1, shuffle=False, num_workers=1)
s_eval_samples = iter(s_dataloader_eval)

l_eval_trajectory_dataset = TrajectoryDataset3(param, 'eval','l')#2
l_dataloader_eval = DataLoader(l_eval_trajectory_dataset, batch_size=1, shuffle=False, num_workers=1)
l_eval_samples = iter(l_dataloader_eval)

r_eval_trajectory_dataset = TrajectoryDataset3(param, 'eval','r')#2
r_dataloader_eval = DataLoader(r_eval_trajectory_dataset, batch_size=1, shuffle=False, num_workers=1)
r_eval_samples = iter(r_dataloader_eval)

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
        cmd = random.sample(['s', 'l', 'r'], 1)[0]
        if cmd == 's':
            batch = next(s_eval_samples)
        elif cmd == 'l':
            batch = next(l_eval_samples)
        else:
            batch = next(r_eval_samples)
            
        batch['img'] = batch['img'].to(device)
        batch['xy'] = batch['xy'].to(device)
        batch['vxy'] = batch['vxy'].to(device)

        output = model(batch['img'], cmd)
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
"""
for j in range(1000):
    eval_error(j)
    draw_traj(j)
"""

while True:
#for index, batch in enumerate(dataloader):
    total_step += 1
    cmd = random.sample(['s', 'l', 'r'], 1)[0]
    if cmd == 's':
        batch = next(s_train_samples)
    elif cmd == 'l':
        batch = next(l_train_samples)
    else:
        batch = next(r_train_samples)
        
    batch['img'] = batch['img'].to(device)
    batch['xy'] = batch['xy'].to(device)
    batch['vxy'] = batch['vxy'].to(device)/opt.max_speed
    
    output = model(batch['img'], cmd)
    output = output.view(opt.batch_size, 10, 4)
    xy = output[:,:,0:2]
    vxy = output[:,:,2:4]

    optimizer.zero_grad()
    #loss_xy = criterion(opt.max_dist*xy, opt.max_dist*batch['xy'])

    loss_xy = 0.5*criterion(xy, batch['xy'])
    #loss_x = criterion(opt.max_dist*xy[:,0], opt.max_dist*batch['xy'][:,0])
    #loss_y = criterion(opt.max_dist*xy[:,1], opt.max_dist*batch['xy'][:,1])
    loss_vxy = 0.5*criterion(vxy, batch['vxy'])

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
    
    if total_step % opt.checkpoint_interval == 0:
        torch.save(model.state_dict(), 'result/saved_models/%s/model_%d.pth'%(opt.dataset_name, total_step))