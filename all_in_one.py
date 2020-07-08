#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import cv2
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
import torchvision.transforms as transforms

from models import GeneratorUNet, Discriminator
from datasets import Yq21Dataset, Yq21Dataset_test, Yq21Dataset_eval

from sensor_manager import SensorManager
from controller import Controller


parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs of training')
parser.add_argument('--dataset_name', type=str, default="test03", help='name of the dataset')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch from which to start lr decay')
parser.add_argument('--n_cpu', type=int, default=16, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_height', type=int, default=128, help='size of image height')
parser.add_argument('--img_width', type=int, default=256, help='size of image width')
parser.add_argument('--channels', type=int, default=3, help='number of image channels') 
opt = parser.parse_args()


random.seed(datetime.now())
torch.manual_seed(999)
torch.cuda.manual_seed(999)
torch.set_num_threads(16)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

generator = GeneratorUNet()
discriminator = Discriminator()

generator = generator.to(device)
generator.load_state_dict(torch.load('result/saved_models/graph/g_71000.pth'))
generator.eval()

img_trans_ = [
        transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]

img_trans = transforms.Compose(img_trans_)

def get_img():
    img = sm['camera'].getImage()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # TODO
    nav = None
    
    img = img_trans(img)
    nav = img_trans(nav)
    
    input_img = torch.cat((img, nav), 0).unsqueeze(0)
    return input_img
    

def get_net_result(input_img):
    with torch.no_grad():
        input_img = input_img.to(device)
        result = generator(input_img)
        return result
        
if __name__ == '__main__':
    sensor_dict = {'camera':None,
                   'lidar':None,
                   }
    sm = SensorManager(sensor_dict)
    sm.init_all()

    ctrl = Controller()
    ctrl.start()
    ctrl.set_forward()

    while True:
        input_img = get_img()
        result = get_net_result(input_img)
        