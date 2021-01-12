#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from os.path import join, dirname
sys.path.insert(0, join(dirname(__file__), '..'))
sys.path.insert(0, join(dirname(__file__), '../../'))
sys.path.insert(0, join(dirname(__file__), '../../../'))

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

from learning.path_model import ModelGRU
from utils import write_params
import carla_utils as cu
from robo_utils.kitti.torch_dataset import TrajectoryDataset

global_trajectory = None
global_trajectory_real = None

random.seed(datetime.now())
torch.manual_seed(233)
torch.cuda.manual_seed(233)
torch.set_num_threads(16)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--test_mode', type=bool, default=False, help='test model switch')
parser.add_argument('--dataset_name', type=str, default="kitti-train-ours-01", help='name of the dataset')
parser.add_argument('--width', type=int, default=400, help='image width')
parser.add_argument('--height', type=int, default=200, help='image height')
parser.add_argument('--scale', type=float, default=30., help='longitudinal length')
parser.add_argument('--batch_size', type=int, default=32, help='size of the batches')
parser.add_argument('--traj_steps', type=int, default=8, help='traj steps')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='adam: weight_decay')
parser.add_argument('--lr', type=float, default=3e-4, help='adam: learning rate')
parser.add_argument('--gamma', type=float, default=0.2, help='xy and vxy loss trade off')
parser.add_argument('--gamma2', type=float, default=0.05, help='xy and axy loss trade off')
parser.add_argument('--img_step', type=int, default=3, help='RNN input image step')
parser.add_argument('--n_cpu', type=int, default=16, help='number of cpu threads to use during batch generation')
parser.add_argument('--checkpoint_interval', type=int, default=2000, help='interval between model checkpoints')
parser.add_argument('--test_interval', type=int, default=10, help='interval between model test')
parser.add_argument('--max_dist', type=float, default=25., help='max distance')
parser.add_argument('--max_speed', type=float, default=10., help='max speed')
parser.add_argument('--max_t', type=float, default=3., help='max time')
opt = parser.parse_args()
if opt.test_mode: opt.batch_size = 1

description = 'kitti train'
log_path = 'result/log/'+opt.dataset_name+'/'
os.makedirs('result/saved_models/%s' % opt.dataset_name, exist_ok=True)
os.makedirs('result/output/%s' % opt.dataset_name, exist_ok=True)
if not opt.test_mode:
    logger = SummaryWriter(log_dir=log_path)
    write_params(log_path, parser, description)
    
model = ModelGRU(256).to(device)
criterion = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

param = cu.parse_yaml_file_unsafe('./param_kitti.yaml')
trajectory_dataset = TrajectoryDataset(param, 'train', opt)
dataloader = DataLoader(trajectory_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=1)

eval_trajectory_dataset = TrajectoryDataset(param, 'eval', opt)
dataloader_eval = DataLoader(eval_trajectory_dataset, batch_size=1, shuffle=False, num_workers=1)
eval_samples = iter(dataloader_eval)


def eval_metric(step):
    model.eval()
    batch = next(eval_samples)
    mask = [2, 5, 8, 10, 13, 17, 20, 23, 26, 29]
    batch['ts_list'] = batch['ts_list'][:,mask]
    batch['x_list'] = batch['x_list'][:,mask]
    batch['y_list'] = batch['y_list'][:,mask]
    batch['vx_list'] = batch['vx_list'][:,mask]
    batch['vy_list'] = batch['vy_list'][:,mask]
    
    t = batch['ts_list'].flatten().unsqueeze(1).to(device)
    t.requires_grad = True
    
    batch['img'] = batch['img'].expand(len(t),10,1,opt.height, opt.width)
    batch['img'] = batch['img'].to(device)
    batch['v_0'] = batch['v_0'].expand(len(t),1)
    batch['v_0'] = batch['v_0'].to(device)
    batch['xy'] = batch['xy'].to(device)
    batch['vxy'] = batch['vxy'].to(device)
    batch['img'].requires_grad = True

    output = model(batch['img'], t, batch['v_0'])
    vx = grad(output[:,0].sum(), t, create_graph=True)[0][:,0]*(opt.max_dist/opt.max_t)
    vy = grad(output[:,1].sum(), t, create_graph=True)[0][:,0]*(opt.max_dist/opt.max_t)

    x = output[:,0]*opt.max_dist
    y = output[:,1]*opt.max_dist

    ax = grad(vx.sum(), t, create_graph=True)[0]*(1./opt.max_t)
    ay = grad(vy.sum(), t, create_graph=True)[0]*(1./opt.max_t)

    jx = grad(ax.sum(), t, create_graph=True)[0]*(1./opt.max_t)
    jy = grad(ay.sum(), t, create_graph=True)[0]*(1./opt.max_t)
    
    vx = vx.data.cpu().numpy()
    vy = vy.data.cpu().numpy()
    x = x.data.cpu().numpy()
    y = y.data.cpu().numpy()
    
    real_x = batch['x_list'].data.cpu().numpy()[0]
    real_y = batch['y_list'].data.cpu().numpy()[0]
    real_vx = batch['vx_list'].data.cpu().numpy()[0]
    real_vy = batch['vy_list'].data.cpu().numpy()[0]
    ts_list = batch['ts_list'].data.cpu().numpy()[0]
    
    ex = np.mean(np.abs(x-real_x))
    ey = np.mean(np.abs(y-real_y))
    evx = np.mean(np.abs(vx - real_vx))
    evy = np.mean(np.abs(vy - real_vy))
    fde = np.hypot(x - real_x, y - real_y)[-1]
    ade = np.mean(np.hypot(x - real_x, y - real_y))
    ev = np.mean(np.hypot(vx - real_vx, vy - real_vy))

    jx = jx.data.cpu().numpy()
    jy = jy.data.cpu().numpy()

    smoothness = np.mean(np.hypot(jx, jy))
    
    logger.add_scalar('metric/ex', ex, step)
    logger.add_scalar('metric/ey', ey, step)
    logger.add_scalar('metric/evx', evx, step)
    logger.add_scalar('metric/evy', evy, step)
    logger.add_scalar('metric/fde', fde, step)
    logger.add_scalar('metric/ade', ade, step)
    logger.add_scalar('metric/ev', ev, step)
    logger.add_scalar('metric/smoothness', smoothness, step)
    model.train()
    
total_step = 0
print('Start to train ...')

for index, batch in enumerate(dataloader):
    total_step += 1

    batch['img'] = batch['img'].to(device)
    batch['t'] = batch['t'].to(device)
    batch['v_0'] = batch['v_0'].to(device)
    batch['xy'] = batch['xy'].to(device)
    batch['vxy'] = batch['vxy'].to(device)
    batch['axy'] = batch['axy'].to(device)
    batch['img'].requires_grad = True
    batch['t'].requires_grad = True

    output = model(batch['img'], batch['t'], batch['v_0'])
    
    vx = grad(output[:,0].sum(), batch['t'], create_graph=True)[0]*(opt.max_dist/opt.max_t)
    vy = grad(output[:,1].sum(), batch['t'], create_graph=True)[0]*(opt.max_dist/opt.max_t)
    output_vxy = torch.cat([vx, vy], dim=1)
    real_v = torch.norm(batch['vxy'], dim=1)
    
    ax = grad(vx.sum(), batch['t'], create_graph=True)[0]
    ay = grad(vy.sum(), batch['t'], create_graph=True)[0]
    output_axy = (1./opt.max_t)*torch.cat([ax, ay], dim=1)

    optimizer.zero_grad()
    loss_xy = criterion(output, batch['xy'])
    loss_vxy = criterion(output_vxy, batch['vxy'])
    loss_axy = criterion(output_axy, batch['axy'])

    loss = loss_xy + opt.gamma*loss_vxy + opt.gamma2*loss_axy

    loss.backward()
    # torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1)
    optimizer.step()

    logger.add_scalar('train/loss_xy', loss_xy.item(), total_step)
    logger.add_scalar('train/loss_vxy', loss_vxy.item(), total_step)
    logger.add_scalar('train/loss_axy', loss_axy.item(), total_step)
    logger.add_scalar('train/loss', loss.item(), total_step)
    
    if total_step % opt.test_interval == 0:
        eval_metric(total_step)
    
    if total_step % opt.checkpoint_interval == 0:
        torch.save(model.state_dict(), 'result/saved_models/%s/model_%d.pth'%(opt.dataset_name, total_step))