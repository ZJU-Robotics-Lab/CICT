#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from os.path import join, dirname
sys.path.insert(0, join(dirname(__file__), '..'))

import os
import time
import random
import argparse
import numpy as np
from PIL import Image, ImageDraw
from datetime import datetime

import torch
from torch.autograd import grad
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from learning.path_model import Model, VAE, CNN_SIN, Model_COS
from learning.costmap_dataset import CostMapDataset
from utils import write_params

random.seed(datetime.now())
torch.manual_seed(666)
torch.cuda.manual_seed(666)
torch.set_num_threads(16)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--test_mode', type=bool, default=False, help='test model switch')
parser.add_argument('--dataset_name', type=str, default="human-data-01", help='name of the dataset')
parser.add_argument('--width', type=int, default=400, help='image width')
parser.add_argument('--height', type=int, default=200, help='image height')
parser.add_argument('--scale', type=float, default=25., help='longitudinal length')
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

description = 'cos model, add v0, final nodes 256, human-data input'
log_path = 'result/log/'+opt.dataset_name+'/'
os.makedirs('result/saved_models/%s' % opt.dataset_name, exist_ok=True)
os.makedirs('result/output/%s' % opt.dataset_name, exist_ok=True)

if not opt.test_mode:
    logger = SummaryWriter(log_dir=log_path)
    write_params(log_path, parser, description)
    
model = Model_COS().to(device)
#model.load_state_dict(torch.load('result/saved_models/human-data-01/model_800000.pth'))
train_loader = DataLoader(CostMapDataset(data_index=[1,2,3,4,6,7,8,9,10,11,12,13,14,15], opt=opt, dataset_path='/media/wang/DATASET/CARLA_HUMAN/town01/'), batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)

eval_loader = DataLoader(CostMapDataset(data_index=[16], opt=opt, dataset_path='/media/wang/DATASET/CARLA_HUMAN/town01/', evalmode=True), batch_size=1, shuffle=False, num_workers=1)
eval_samples = iter(eval_loader)

criterion = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

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
    
def draw_traj():
    model.eval()
    batch = next(eval_samples)
    img = batch['img'].clone().data.numpy().squeeze()*127+128
    
    t = torch.arange(0, 0.99, 0.01).unsqueeze(1).to(device)
    t.requires_grad = True
    
    batch['img'] = batch['img'].expand(len(t),1,opt.height, opt.width)
    batch['img'] = batch['img'].to(device)
    batch['t'] = batch['t'].to(device)
    batch['v_0'] = batch['v_0'].expand(len(t),1)
    batch['v_0'] = batch['v_0'].to(device)
    batch['xy'] = batch['xy'].to(device)
    batch['vxy'] = batch['vxy'].to(device)
    batch['img'].requires_grad = True
    batch['t'].requires_grad = True
    
    output = model(batch['img'], t, batch['v_0'])
    #output = model(batch['img'], t)
    
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
    
    img.save(('result/output/%s/' % opt.dataset_name)+str(time.time())+'.png')
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

total_step = 0
if opt.test_mode:
    abs_x = []
    abs_y = []
    
print('Start to train ...')
for i, batch in enumerate(train_loader):
    total_step += 1
    if opt.test_mode:
        for j in range(1000): draw_traj()
        break

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

    optimizer.zero_grad()
    loss_xy = criterion(output, batch['xy'])
    loss_vxy = criterion(output_vxy, batch['vxy'])
    #KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    loss = loss_xy + opt.gamma*loss_vxy# + opt.gamma2*KLD
    
    loss.backward()
    torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1)
    optimizer.step()

    logger.add_scalar('train/loss_xy', opt.max_dist*loss_xy.item(), total_step)
    logger.add_scalar('train/loss_vxy', loss_vxy.item(), total_step)
    #logger.add_scalar('train/loss_kld', KLD.item(), total_step)
    logger.add_scalar('train/loss', loss.item(), total_step)

    if total_step % opt.test_interval == 0:
        eval_error(total_step)
    
    if total_step % opt.checkpoint_interval == 0:
        torch.save(model.state_dict(), 'result/saved_models/%s/model_%d.pth'%(opt.dataset_name, total_step))