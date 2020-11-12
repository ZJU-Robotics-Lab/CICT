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
from utils import write_params, fig2data

import carla_utils as cu
from learning.robo_dataset_utils.robo_utils.kitti.torch_dataset import TrajectoryDataset

global_trajectory = None
global_trajectory_real = None

random.seed(datetime.now())
torch.manual_seed(233)
torch.cuda.manual_seed(233)
torch.set_num_threads(16)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--test_mode', type=bool, default=False, help='test model switch')
parser.add_argument('--dataset_name', type=str, default="kitti-train-ours-04", help='name of the dataset')
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
parser.add_argument('--test_interval', type=int, default=50, help='interval between model test')
parser.add_argument('--max_dist', type=float, default=25., help='max distance')
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
    
model = ModelGRU(256).to(device)
#model.load_state_dict(torch.load('result/saved_models/kitti-train-ours-01/model_396000.pth'))
criterion = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

param = cu.parse_yaml_file_unsafe('../../learning/robo_dataset_utils/params/param_kitti.yaml')
trajectory_dataset = TrajectoryDataset(param, 'train')#7
dataloader = DataLoader(trajectory_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)

eval_trajectory_dataset = TrajectoryDataset(param, 'eval')#2
dataloader_eval = DataLoader(eval_trajectory_dataset, batch_size=1, shuffle=False, num_workers=opt.n_cpu)
eval_samples = iter(dataloader_eval)

def xy2uv(x, y):
    y = -y
    pixs_per_meter = opt.height/opt.scale
    u = (opt.height-x*pixs_per_meter-1).astype(int)
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
    max_speed = 15.0

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
    
    t = max_x*np.arange(0, 1.0, 1./x.shape[0])
    real_t = max_x*global_trajectory_real['ts_list']
    vx = trajectory['vx']
    vy = trajectory['vy']
    real_vx = real_trajectory['vx']
    real_vy = real_trajectory['vy']
    v = np.sqrt(np.power(vx, 2), np.power(vy, 2))
    angle = np.arctan2(vy, vx)/np.pi*max_speed
    real_v = np.sqrt(np.power(real_vx, 2), np.power(real_vy, 2))

    real_angle = np.arctan2(real_vy, real_vx)/np.pi*max_speed
    ax2 = ax1.twinx()
    #ax2.plot(real_t, real_vx, label='vx', color = 'tab:cyan', linewidth=2)
    #ax2.plot(real_t, real_vy, label='vy', color = 'tab:purple', linewidth=2)
    ax2.plot(t, v, label='speed', color = 'r', linewidth=2)
    ax2.plot(t, angle, label='angle', color = 'g', linewidth=2)
    # real
    ax2.plot(real_t, real_v, label='real-speed', color = 'r', linewidth=2, linestyle='--')
    ax2.plot(real_t, real_angle, label='real-angle', color = 'g', linewidth=2, linestyle='--')
    
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

    x = output[:,0]*opt.max_dist
    y = output[:,1]*opt.max_dist
    
    vx = vx.data.cpu().numpy()
    vy = vy.data.cpu().numpy()
    x = x.data.cpu().numpy()
    y = y.data.cpu().numpy()
    
    real_x = batch['x_list'].data.cpu().numpy()[0]
    real_y = batch['y_list'].data.cpu().numpy()[0]
    real_vx = batch['vx_list'].data.cpu().numpy()[0]
    real_vy = batch['vy_list'].data.cpu().numpy()[0]
    ts_list = batch['ts_list'].data.cpu().numpy()[0]

    global_trajectory = {'x':x, 'y':y, 'vx':vx, 'vy':vy}
    global_trajectory_real = {'x':real_x, 'y':real_y, 'vx':real_vx, 'vy':real_vy, 'ts_list':ts_list}
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

def eval_metric(step):
    global global_trajectory, global_trajectory_real
    
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
    #print('batch[xy]', batch['xy'])
    #print(t.shape, batch['ts_list'].flatten().unsqueeze(1).shape)
    
    output = model(batch['img'], t, batch['v_0'])
    vx = grad(output[:,0].sum(), t, create_graph=True)[0][:,0]*(opt.max_dist/opt.max_t)
    vy = grad(output[:,1].sum(), t, create_graph=True)[0][:,0]*(opt.max_dist/opt.max_t)

    x = output[:,0]*opt.max_dist
    y = output[:,1]*opt.max_dist
    
    vx = vx.data.cpu().numpy()
    vy = vy.data.cpu().numpy()
    x = x.data.cpu().numpy()
    y = y.data.cpu().numpy()
    
    real_x = batch['x_list'].data.cpu().numpy()[0]
    real_y = batch['y_list'].data.cpu().numpy()[0]
    real_vx = batch['vx_list'].data.cpu().numpy()[0]
    real_vy = batch['vy_list'].data.cpu().numpy()[0]
    #ts_list = batch['ts_list'].data.cpu().numpy()[0]*opt.max_t
    
    ex = np.mean(np.abs(x-real_x))
    ey = np.mean(np.abs(y-real_y))
    evx = np.mean(np.abs(vx - real_vx))
    evy = np.mean(np.abs(vy - real_vy))
    fde = np.abs(x-real_x)[-1]
    ade = np.mean(np.hypot(x - real_x, y - real_y))
    ev = np.mean(np.hypot(vx - real_vx, vy - real_vy))
    
    logger.add_scalar('metric/ex', ex, step)
    logger.add_scalar('metric/ey', ey, step)
    logger.add_scalar('metric/evx', evx, step)
    logger.add_scalar('metric/evy', evy, step)
    logger.add_scalar('metric/fde', fde, step)
    logger.add_scalar('metric/ade', ade, step)
    logger.add_scalar('metric/ev', ev, step)
    
    #global_trajectory = {'x':x, 'y':y, 'vx':vx, 'vy':vy}
    #global_trajectory_real = {'x':real_x, 'y':real_y, 'vx':real_vx, 'vy':real_vy, 'ts_list':ts_list}
    
    model.train()
    
    
def eval_error(total_step):
    model.eval()
    abs_x = []
    abs_y = []
    abs_vx = []
    abs_vy = []
    abs_ax = []
    abs_ay = []
    #abs_a = []
    rel_x = []
    rel_y = []
    rel_vx = []
    rel_vy = []
    rel_ax = []
    rel_ay = []
    #rel_a = []
    for i in range(100):
        batch = next(eval_samples)
        batch['img'] = batch['img'].to(device)
        batch['t'] = batch['t'].to(device)
        batch['v_0'] = batch['v_0'].to(device)
        batch['xy'] = batch['xy'].to(device)
        batch['vxy'] = batch['vxy'].to(device)
        batch['axy'] = batch['axy'].to(device)
        batch['img'].requires_grad = True
        batch['t'].requires_grad = True
    
        #output = model(batch['img'], batch['t'])
        output = model(batch['img'], batch['t'], batch['v_0'])
        vx = (opt.max_dist/opt.max_t)*grad(output[:,0].sum(), batch['t'], create_graph=True)[0]
        vy = (opt.max_dist/opt.max_t)*grad(output[:,1].sum(), batch['t'], create_graph=True)[0]
        
        ax = grad(vx.sum(), batch['t'], create_graph=True)[0]
        ay = grad(vy.sum(), batch['t'], create_graph=True)[0]
        #output_axy = (1./opt.max_t)*torch.cat([ax, ay], dim=1)
        
        #theta_a = torch.atan2(ay, ax)
        #theta_v = torch.atan2(vy, vx)
        #sign = torch.sign(torch.cos(theta_a-theta_v)).detach()
        #a = torch.mul(torch.norm(output_axy, dim=1), sign.flatten()).unsqueeze(1)
    
        gen = output.data.cpu().numpy()[0]
        real = batch['xy'].data.cpu().numpy()[0]
        vx = vx.data.cpu().numpy()[0][0]
        vy = vy.data.cpu().numpy()[0][0]
        ax = ax.data.cpu().numpy()[0][0]
        ay = ay.data.cpu().numpy()[0][0]
        #a = a.data.cpu().numpy()[0][0]
        real_v = batch['vxy'].data.cpu().numpy()[0]
        real_a = batch['axy'].data.cpu().numpy()[0]
        #real_a = batch['a'].data.cpu().numpy()[0]
        abs_x.append(abs(gen[0]-real[0]))
        abs_y.append(abs(gen[1]-real[1]))
        abs_vx.append(abs(vx - real_v[0]))
        abs_vy.append(abs(vy - real_v[1]))
        #abs_a.append(abs(a - real_a))
        abs_ax.append(abs(ax - real_a[0]))
        abs_ay.append(abs(ay - real_a[1]))
        if abs(real[0]) > 0.05:
            rel_x.append(abs(gen[0]-real[0])/(abs(real[0])))
        if abs(real[1]) > 0.05:
            rel_y.append(abs(gen[1]-real[1])/(abs(real[1])))
        if abs(real_v[0]) > 0.05:
            rel_vx.append(abs(vx - real_v[0])/(abs(real_v[0])))
        if abs(real_v[1]) > 0.05:
            rel_vy.append(abs(vy - real_v[1])/(abs(real_v[1])))
        if abs(real_a[0]) > 0.05:
            rel_ax.append(abs(ax - real_a[0])/(abs(real_a[0])))
        if abs(real_a[1]) > 0.05:
            rel_ay.append(abs(ay - real_a[1])/(abs(real_a[1])))
        #if abs(real_a) > 0.05:
        #    rel_a.append(abs(a - real_a)/(abs(real_a)))

    logger.add_scalar('eval/x', opt.max_dist*sum(abs_x)/len(abs_x), total_step)
    logger.add_scalar('eval/y', opt.max_dist*sum(abs_y)/len(abs_y), total_step)
    logger.add_scalar('eval/vx', sum(abs_vx)/len(abs_vx), total_step)
    logger.add_scalar('eval/vy', sum(abs_vy)/len(abs_vy), total_step)
    logger.add_scalar('eval/ax', sum(abs_ax)/len(abs_ax), total_step)
    logger.add_scalar('eval/ay', sum(abs_ay)/len(abs_ay), total_step)
    #logger.add_scalar('eval/a', sum(abs_a)/len(abs_a), total_step)

    if len(rel_x) > 0:
        logger.add_scalar('eval/rel_x', sum(rel_x)/len(rel_x), total_step)
    if len(rel_y) > 0:
        logger.add_scalar('eval/rel_y', sum(rel_y)/len(rel_y), total_step)
    if len(rel_vx) > 0:
        logger.add_scalar('eval/rel_vx', sum(rel_vx)/len(rel_vx), total_step)
    if len(rel_vy) > 0:
        logger.add_scalar('eval/rel_vy', sum(rel_vy)/len(rel_vy), total_step)
    if len(rel_ax) > 0:
        logger.add_scalar('eval/rel_ax', sum(rel_ax)/len(rel_ax), total_step)
    if len(rel_ay) > 0:
        logger.add_scalar('eval/rel_ay', sum(rel_ay)/len(rel_ay), total_step)
    #if len(rel_a) > 0:
    #    logger.add_scalar('eval/rel_a', sum(rel_a)/len(rel_a), total_step)
    model.train()
    
total_step = 0
print('Start to train ...')
"""
for j in range(1000): eval_metric(j)
"""
for index, batch in enumerate(dataloader):
    total_step += 1
    if opt.test_mode:
        for j in range(500): draw_traj(j)
        break

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
    #v = torch.norm(output_vxy, dim=1)
    real_v = torch.norm(batch['vxy'], dim=1)
    
    ax = grad(vx.sum(), batch['t'], create_graph=True)[0]
    ay = grad(vy.sum(), batch['t'], create_graph=True)[0]
    output_axy = (1./opt.max_t)*torch.cat([ax, ay], dim=1)
    
    #theta_a = torch.atan2(ay, ax)
    #theta_v = torch.atan2(vy, vx)
    #sign = torch.sign(torch.cos(theta_a-theta_v)).detach()
    #a = torch.mul(torch.norm(output_axy, dim=1), sign.flatten()).unsqueeze(1)

    optimizer.zero_grad()
    loss_xy = criterion(opt.max_dist*output, opt.max_dist*batch['xy'])
    # loss_vxy = criterion(v, real_v)
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
        #eval_error(total_step)
        #draw_traj(total_step)
        eval_metric(total_step)
    
    if total_step % opt.checkpoint_interval == 0:
        torch.save(model.state_dict(), 'result/saved_models/%s/model_%d.pth'%(opt.dataset_name, total_step))