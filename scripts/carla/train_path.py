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

from learning.path_model import ModelGRU
from learning.costmap_dataset import CostMapDataset
from utils import write_params

global_trajectory = None
global_trajectory_real = None

random.seed(datetime.now())
torch.manual_seed(666)
torch.cuda.manual_seed(666)
torch.set_num_threads(16)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--test_mode', type=bool, default=False, help='test model switch')
parser.add_argument('--dataset_name', type=str, default="train-01", help='name of the dataset')
parser.add_argument('--width', type=int, default=400, help='image width')
parser.add_argument('--height', type=int, default=200, help='image height')
parser.add_argument('--scale', type=float, default=25., help='longitudinal length')
parser.add_argument('--batch_size', type=int, default=32, help='size of the batches')
parser.add_argument('--traj_steps', type=int, default=8, help='traj steps')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='adam: weight_decay')
parser.add_argument('--lr', type=float, default=3e-4, help='adam: learning rate')
parser.add_argument('--gamma', type=float, default=0.1, help='xy and vxy loss trade off')
parser.add_argument('--gamma2', type=float, default=0.001, help='xy and axy loss trade off')
parser.add_argument('--img_step', type=int, default=3, help='RNN input image step')
parser.add_argument('--n_cpu', type=int, default=16, help='number of cpu threads to use during batch generation')
parser.add_argument('--checkpoint_interval', type=int, default=2000, help='interval between model checkpoints')
parser.add_argument('--test_interval', type=int, default=200, help='interval between model test')
parser.add_argument('--max_dist', type=float, default=25., help='max distance')
parser.add_argument('--max_t', type=float, default=3., help='max time')
opt = parser.parse_args()
if opt.test_mode: opt.batch_size = 1
    
description = 'train'
log_path = 'result/log/'+opt.dataset_name+'/'
os.makedirs('result/saved_models/%s' % opt.dataset_name, exist_ok=True)
os.makedirs('result/output/%s' % opt.dataset_name, exist_ok=True)

logger = SummaryWriter(log_dir=log_path)
write_params(log_path, parser, description)
    
model = ModelGRU(256).to(device)

train_loader = DataLoader(CostMapDataset(data_index=[1,2,3,4,5,6,7,9,10], opt=opt, dataset_path='/media/wang/DATASET/CARLA_HUMAN/town01/'), batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)

test_loader = DataLoader(CostMapDataset(data_index=[8], opt=opt, dataset_path='/media/wang/DATASET/CARLA_HUMAN/town01/', evalmode=True), batch_size=1, shuffle=False, num_workers=1)
test_samples = iter(test_loader)

criterion = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
 

def eval_error(total_step):
    model.eval()
    abs_x = []
    abs_y = []
    abs_vx = []
    abs_vy = []
    abs_a = []

    for i in range(100):
        batch = next(test_samples)
        batch['img'] = batch['img'].to(device)
        batch['t'] = batch['t'].to(device)
        batch['v_0'] = batch['v_0'].to(device)
        batch['xy'] = batch['xy'].to(device)
        batch['vxy'] = batch['vxy'].to(device)
        batch['axy'] = batch['axy'].to(device)
        batch['img'].requires_grad = True
        batch['t'].requires_grad = True
    
        output = model(batch['img'], batch['t'], batch['v_0'])
        vx = (opt.max_dist/opt.max_t)*grad(output[:,0].sum(), batch['t'], create_graph=True)[0]
        vy = (opt.max_dist/opt.max_t)*grad(output[:,1].sum(), batch['t'], create_graph=True)[0]
        
        ax = grad(vx.sum(), batch['t'], create_graph=True)[0]
        ay = grad(vy.sum(), batch['t'], create_graph=True)[0]
        output_axy = (1./opt.max_t)*torch.cat([ax, ay], dim=1)
        
        theta_a = torch.atan2(ay, ax)
        theta_v = torch.atan2(vy, vx)
        sign = torch.sign(torch.cos(theta_a-theta_v)).detach()
        a = torch.mul(torch.norm(output_axy, dim=1), sign.flatten()).unsqueeze(1)
    
        gen = output.data.cpu().numpy()[0]
        real = batch['xy'].data.cpu().numpy()[0]
        vx = vx.data.cpu().numpy()[0][0]
        vy = vy.data.cpu().numpy()[0][0]
        a = a.data.cpu().numpy()[0][0]
        real_v = batch['vxy'].data.cpu().numpy()[0]
        #real_a = batch['axy'].data.cpu().numpy()[0]
        real_a = batch['a'].data.cpu().numpy()[0]
        abs_x.append(abs(gen[0]-real[0]))
        abs_y.append(abs(gen[1]-real[1]))
        abs_vx.append(abs(vx - real_v[0]))
        abs_vy.append(abs(vy - real_v[1]))
        abs_a.append(abs(a - real_a))

    logger.add_scalar('eval/x', opt.max_dist*sum(abs_x)/len(abs_x), total_step)
    logger.add_scalar('eval/y', opt.max_dist*sum(abs_y)/len(abs_y), total_step)
    logger.add_scalar('eval/vx', sum(abs_vx)/len(abs_vx), total_step)
    logger.add_scalar('eval/vy', sum(abs_vy)/len(abs_vy), total_step)
    logger.add_scalar('eval/a', sum(abs_a)/len(abs_a), total_step)

    model.train()


total_step = 0
print('Start to train ...')
for i, batch in enumerate(train_loader):
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
    
    vx = grad(output[:,0].sum(), batch['t'], create_graph=True)[0]
    vy = grad(output[:,1].sum(), batch['t'], create_graph=True)[0]
    output_vxy = (opt.max_dist/opt.max_t)*torch.cat([vx, vy], dim=1)
    
    ax = grad(vx.sum(), batch['t'], create_graph=True)[0]
    ay = grad(vy.sum(), batch['t'], create_graph=True)[0]
    output_axy = (1./opt.max_t)*torch.cat([ax, ay], dim=1)
    
    optimizer.zero_grad()
    loss_xy = criterion(output, batch['xy'])
    loss_vxy = criterion(output_vxy, batch['vxy'])
    loss_axy = criterion(output_axy, batch['axy'])
    loss = loss_xy + opt.gamma*loss_vxy + opt.gamma2*loss_axy

    loss.backward()
    torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1)
    optimizer.step()


    logger.add_scalar('train/loss_xy', loss_xy.item(), total_step)
    logger.add_scalar('train/loss_vxy', loss_vxy.item(), total_step)
    logger.add_scalar('train/loss_axy', loss_axy.item(), total_step)
    logger.add_scalar('train/loss', loss.item(), total_step)
    
    if total_step % opt.test_interval == 0:
        eval_error(total_step)
    
    if total_step % opt.checkpoint_interval == 0:
        torch.save(model.state_dict(), 'result/saved_models/%s/model_%d.pth'%(opt.dataset_name, total_step))