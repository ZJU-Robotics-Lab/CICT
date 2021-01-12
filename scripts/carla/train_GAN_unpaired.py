#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from os.path import join, dirname
sys.path.insert(0, join(dirname(__file__), '..'))
sys.path.insert(0, join(dirname(__file__), '../../'))

import os
import random
import argparse
import numpy as np
from PIL import Image
from datetime import datetime

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

from learning.models import GeneratorUNet, Discriminator
from learning.costmap_dataset import CARLADataset
from utils import write_params

random.seed(datetime.now())
torch.manual_seed(999)
torch.cuda.manual_seed(999)
torch.set_num_threads(16)

parser = argparse.ArgumentParser()
parser.add_argument('--eval', type=bool, default=False, help='if eval')
parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs of training')
parser.add_argument('--dataset_name', type=str, default="cgan-human-data-01", help='name of the dataset')
parser.add_argument('--batch_size', type=int, default=24, help='size of the batches')
parser.add_argument('--lr', type=float, default=3e-4, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch from which to start lr decay')
parser.add_argument('--n_cpu', type=int, default=16, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_height', type=int, default=128, help='size of image height')
parser.add_argument('--img_width', type=int, default=256, help='size of image width')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=100, help='interval between sampling of images from generators')
parser.add_argument('--checkpoint_interval', type=int, default=1000, help='interval between model checkpoints')
opt = parser.parse_args()
#print(opt)

description = 'cgan train, dynamic obstacles'
log_path = 'result/log/'+opt.dataset_name+'/'
os.makedirs('result/images/%s' % opt.dataset_name, exist_ok=True)
os.makedirs('result/saved_models/%s' % opt.dataset_name, exist_ok=True)

if not opt.eval:
    logger = SummaryWriter(log_dir=log_path)
    write_params(log_path, parser, description)
    

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Loss functions
criterion_GAN = torch.nn.MSELoss()
criterion_pixelwise = torch.nn.L1Loss()

# Loss weight of L1 pixel-wise loss between translated image and real image
lambda_pixel = 100
# Calculate output of image discriminator (PatchGAN)
patch = (1, opt.img_height//2**4, opt.img_width//2**4)

generator = GeneratorUNet()
discriminator = Discriminator()

generator = generator.to(device)
discriminator = discriminator.to(device)
#generator.load_state_dict(torch.load('../../ckpt/sim/g.pth'))
#discriminator.load_state_dict(torch.load('../../ckpt/sim/d.pth'))

criterion_GAN.to(device)
criterion_pixelwise.to(device)
#unet encode

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

if opt.eval:
    eval_dataloader = DataLoader(CARLADataset(data_index=[16], eval_mode=True),batch_size=1, shuffle=False, num_workers=1)
    eval_samples = iter(eval_dataloader)
    
def eval_dataset():
    steps = 0
    generator.eval()
    with torch.no_grad():
        for i in range(len(eval_dataloader)):
            test_batch = next(eval_samples)
            test_a1 = test_batch['A1']
            test_a2 = test_batch['A2']
            test_real_b = test_batch['B']
            file_name = test_batch['file_name'][0]
            test_real_a = torch.cat((test_a1, test_a2), 1)
        
            test_real_a = test_real_a.to(device)
            test_real_b = test_real_b.to(device)
            test_a1 = test_a1.to(device)
            test_a2 = test_a2.to(device)
            test_fake_b = generator(test_real_a)

            save_image(test_a1.data, 'result/images/%s/%s_img.png' % (opt.dataset_name, file_name), nrow=1, normalize=True)
            save_image(test_a2.data, 'result/images/%s/%s_nav.png' % (opt.dataset_name, file_name), nrow=1, normalize=True)
            save_image(test_real_b.data, 'result/images/%s/%s_real.png' % (opt.dataset_name, file_name), nrow=1, normalize=True)
            save_image(test_fake_b.data, 'result/images/%s/%s_fake.png' % (opt.dataset_name, file_name), nrow=1, normalize=True)
            steps += 1
    
    generator.train()

train_loader = DataLoader(CARLADataset(data_index=[1,2,3,4,5,6,7,8]),
                            batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)
test_val_dataloader = DataLoader(CARLADataset(data_index=[9], eval_mode=True),
                            batch_size=opt.n_cpu, shuffle=False, num_workers=opt.n_cpu)
test_samples = iter(test_val_dataloader)

def sample_images(steps):
    #Saves a generated sample from the validation set
    generator.eval()
    with torch.no_grad():
        test_batch = next(test_samples)
        test_a1 = test_batch['A1']
        test_a2 = test_batch['A2']
        test_real_b = test_batch['B']
    
        test_real_a = torch.cat((test_a1, test_a2), 1)
    
        test_real_a = Variable(test_real_a).to(device)
        test_real_b = Variable(test_real_b).to(device)
        test_a1 = Variable(test_a1).to(device)
        test_a2 = Variable(test_a2).to(device)
        test_fake_b = generator(test_real_a)
    
        img_sample = torch.cat((test_a1.data, test_a2.data), -2)
        pre_sample = torch.cat((test_fake_b.data, test_real_b.data), -2)
        #pre_sample = test_fake_b
    
        save_image(img_sample, 'result/images/%s/%s_img.png' % (opt.dataset_name, steps), nrow=4, normalize=True)
        save_image(pre_sample, 'result/images/%s/%s_pre.png' % (opt.dataset_name, steps), nrow=4, normalize=True)
        #save_image(test_fake_b.data, 'result/images/%s/%s_fake.png' % (opt.dataset_name, steps), nrow=4, normalize=True)
        #logger.add_image('img/origin', img_sample[0])
        #logger.add_image('img/prediction', pre_sample[0]*128)
    
    generator.train()

#eval_dataset()

print('Start to train ...')
total_step = 0
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(train_loader):
        total_step += 1
        real_A = batch['A']
        real_B = batch['B']
        fake_nav = batch['fake_nav_with_img']
        valid = Variable(torch.from_numpy(np.ones((real_B.size(0), *patch))).float(), requires_grad=False).to(device)
        fake = Variable(torch.from_numpy(np.zeros((real_B.size(0), *patch))).float(), requires_grad=False).to(device)
        # Model inputs
        real_A = Variable(real_A).to(device)
        real_B = Variable(real_B).to(device)
        fake_nav = Variable(fake_nav).to(device)
        fake_B = generator(real_A)
        fake_B2 = generator(fake_nav)
        # GAN loss
        pred_fake = discriminator(fake_B, real_A)
        pred_fake2 = discriminator(fake_B2, fake_nav)
        
        optimizer_G.zero_grad()
        loss_GAN = criterion_GAN(pred_fake, valid)
        loss_GAN2 = criterion_GAN(pred_fake2, valid)
        # Pixel-wise loss
        loss_pixel = criterion_pixelwise(fake_B, real_B)
        # Total loss
        loss_G = (loss_GAN + loss_GAN2)*0.5 + lambda_pixel*loss_pixel
        #loss_G = loss_pixel
        loss_G.backward()
        torch.nn.utils.clip_grad_value_(generator.parameters(), clip_value=20)
        optimizer_G.step()
        
        optimizer_D.zero_grad()

        # Real loss
        pred_real = discriminator(real_B, real_A)
        loss_real = criterion_GAN(pred_real, valid)
        # Fake loss
        pred_fake = discriminator(fake_B.detach(), real_A)
        pred_fake2 = discriminator(fake_B2.detach(), fake_nav)
        loss_fake = criterion_GAN(pred_fake, fake)
        loss_fake2 = criterion_GAN(pred_fake2, fake)
        # Total loss
        loss_D = 0.5 * (loss_real + (loss_fake+loss_fake2)*0.5)
        loss_D.backward()
        torch.nn.utils.clip_grad_value_(discriminator.parameters(), clip_value=20)
        optimizer_D.step()

        logger.add_scalar('loss/loss_D', loss_D.item(), total_step)
        logger.add_scalar('loss/loss_G', loss_G.item(), total_step)
        logger.add_scalar('loss/loss_pixel', loss_pixel.item(), total_step)
        logger.add_scalar('loss/loss_GAN', loss_GAN.item(), total_step)

        # If at sample interval save image
        if total_step % opt.sample_interval == 0:
            sample_images(total_step)


        #if total_step % 500 == 0:
        #    for name, param in discriminator.named_parameters():
        #        logger.add_histogram('discriminator/'+name, param.clone().cpu().data.numpy(), total_step)
        #    for name, param in generator.named_parameters():
        #        logger.add_histogram('generator/'+name, param.clone().cpu().data.numpy(), total_step)

        if opt.checkpoint_interval != -1 and total_step % opt.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(generator.state_dict(), 'result/saved_models/%s/g_%d.pth' %
                       (opt.dataset_name, total_step))
            torch.save(discriminator.state_dict(), 'result/saved_models/%s/d_%d.pth'
                       % (opt.dataset_name, total_step))