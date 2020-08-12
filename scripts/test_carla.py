#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from os.path import join, dirname
sys.path.insert(0, join(dirname(__file__), '..'))

import simulator
simulator.load('/home/wang/CARLA_0.9.9.4')
import carla
sys.path.append('/home/wang/CARLA_0.9.9.4/PythonAPI/carla')
from agents.navigation.basic_agent import BasicAgent

from simulator import config, set_weather, add_vehicle
from simulator.sensor_manager import SensorManager
from utils.navigator_sim import get_random_destination, get_map, get_nav, replan, close2dest
from learning.models import GeneratorUNet
from learning.path_model import Model_COS

from ff_collect_pm_data import sensor_dict
from ff.collect_ipm import InversePerspectiveMapping
from ff.carla_sensor import Sensor, CarlaSensorMaster

import os
import cv2
import time
import copy
import random
import argparse
import numpy as np
from PIL import Image, ImageDraw
from datetime import datetime

import torch
from torch.autograd import grad
import torchvision.transforms as transforms

global_img = None
global_pcd = None
global_nav = None
global_vel = 0.

MAX_SPEED = 20
img_height = 128
img_width = 256
longitudinal_length = 25.0 # [m]

random.seed(datetime.now())
torch.manual_seed(999)
torch.cuda.manual_seed(999)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

generator = GeneratorUNet()
generator = generator.to(device)
generator.load_state_dict(torch.load('../ckpt/sim/g.pth'))
model = Model_COS().to(device)
model.load_state_dict(torch.load('../ckpt/sim/model.pth'))
generator.eval()
model.eval()

parser = argparse.ArgumentParser(description='Params')
parser.add_argument('-d', '--data', type=int, default=1, help='data index')
parser.add_argument('-n', '--num', type=int, default=100000, help='total number')
parser.add_argument('--width', type=int, default=400, help='image width')
parser.add_argument('--height', type=int, default=200, help='image height')
parser.add_argument('--max_dist', type=float, default=20., help='max distance')
parser.add_argument('--max_t', type=float, default=5., help='max time')
parser.add_argument('--scale', type=float, default=25., help='longitudinal length')
args = parser.parse_args()

data_index = args.data
save_path = '/media/wang/DATASET/CARLA/town01/'+str(data_index)+'/'

img_transforms = [
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
img_trans = transforms.Compose(img_transforms)

cost_map_transforms_ = [ transforms.Resize((200, 400), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
    ]
cost_map_trans = transforms.Compose(cost_map_transforms_)
        
class Param(object):
    def __init__(self):
        self.longitudinal_length = longitudinal_length
        self.ksize = 21
        
param = Param()
sensor = Sensor(sensor_dict['camera']['transform'], config['camera'])
sensor_master = CarlaSensorMaster(sensor, sensor_dict['camera']['transform'], binded=True)

def get_cost_map(img, point_cloud):
    img2 = np.zeros((args.height, args.width), np.uint8)
    img2.fill(255)
    
    pixs_per_meter = args.height/longitudinal_length
    u = (args.height-point_cloud[0]*pixs_per_meter).astype(int)
    v = (-point_cloud[1]*pixs_per_meter+args.width//2).astype(int)
    
    mask = np.where((u >= 0)&(u < args.height))[0]
    u = u[mask]
    v = v[mask]
    
    mask = np.where((v >= 0)&(v < args.width))[0]
    u = u[mask]
    v = v[mask]

    img2[u,v] = 0
    
    kernel = np.ones((17,17),np.uint8)  
    img2 = cv2.erode(img2,kernel,iterations = 1)
    
    img = cv2.addWeighted(img,0.7,img2,0.3,0)
    kernel_size = (17, 17)
    sigma = 21
    img = cv2.GaussianBlur(img, kernel_size, sigma);
    return img

def xy2uv(x, y):
    pixs_per_meter = args.height/args.scale
    u = (args.height-x*pixs_per_meter).astype(int)
    v = (y*pixs_per_meter+args.width//2).astype(int)
    return u, v

def draw_traj(cost_map):
    global global_vel
    img = Image.fromarray(cv2.cvtColor(cost_map,cv2.COLOR_BGR2RGB)).convert('L')
    img = cost_map_trans(img)
    
    t = torch.arange(0, 0.7, 0.01).unsqueeze(1).to(device)
    t.requires_grad = True
    
    img = img.expand(len(t),1,args.height, args.width)
    img = img.to(device)
    img.requires_grad = True

    v_0 = torch.FloatTensor([global_vel]).expand(len(t),1)
    v_0 = v_0.to(device)

    output = model(img, t, v_0)
    
    cost_map = Image.fromarray(cost_map).convert("RGB")
    draw = ImageDraw.Draw(cost_map)

    result = output.data.cpu().numpy()
    x = args.max_dist*result[:,0]
    y = args.max_dist*result[:,1]
    u, v = xy2uv(x, y)

    for i in range(len(u)-1):
        draw.line((v[i], u[i], v[i+1], u[i+1]), 'red')
        draw.line((v[i]+1, u[i], v[i+1]+1, u[i+1]), 'red')
        draw.line((v[i]-1, u[i], v[i+1]-1, u[i+1]), 'red')
        
    return cv2.cvtColor(np.asarray(cost_map),cv2.COLOR_RGB2BGR)

def get_traj(cost_map, t, show=False):
    global global_vel
    img = Image.fromarray(cv2.cvtColor(cost_map,cv2.COLOR_BGR2RGB)).convert('L')
    img = cost_map_trans(img)
    
    t = torch.FloatTensor([t/args.max_t]).unsqueeze(1).to(device)
    t.requires_grad = True
    
    img = img.expand(len(t),1,args.height, args.width)
    img = img.to(device)
    img.requires_grad = True

    v_0 = torch.FloatTensor([global_vel]).expand(len(t),1)
    v_0 = v_0.to(device)

    output = model(img, t, v_0)
    vx = (args.max_dist/args.max_t)*grad(output[:,0].sum(), t, create_graph=True)[0].data.cpu().numpy()[0][0]
    vy = (args.max_dist/args.max_t)*grad(output[:,1].sum(), t, create_graph=True)[0].data.cpu().numpy()[0][0]
    
    ######################
    x = output.data.cpu().numpy()[0][0]*args.max_dist
    y = output.data.cpu().numpy()[0][1]*args.max_dist
    trajectory = {'x':x, 'y':y, 'vx':vx, 'vy':vy}
    
    ####################
    if show:
        cost_map = Image.fromarray(cost_map).convert("RGB")
    
        result = output.data.cpu().numpy()
        x = args.max_dist*result[:,0]
        y = args.max_dist*result[:,1]
        u, v = xy2uv(x, y)
        u = u[0]
        v = v[0]
        r = 10
        draw = ImageDraw.Draw(cost_map)
        draw.ellipse((max(0,v-r), max(0,u-r), v+r, u+r), fill='red', outline='red', width=10)
    
    return cv2.cvtColor(np.asarray(cost_map),cv2.COLOR_RGB2BGR), trajectory

def mkdir(path):
    os.makedirs(save_path+path, exist_ok=True)

def image_callback(data):
    global global_img
    array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8")) 
    array = np.reshape(array, (data.height, data.width, 4)) # RGBA format
    global_img = array
    
def lidar_callback(data):
    global global_pcd
    lidar_data = np.frombuffer(data.raw_data, dtype=np.float32).reshape([-1, 3])
    point_cloud = np.stack([-lidar_data[:,1], -lidar_data[:,0], -lidar_data[:,2]])
    mask = np.where((point_cloud[0] > 1.0)|(point_cloud[0] < -4.0)|(point_cloud[1] > 1.2)|(point_cloud[1] < -1.2))[0]
    point_cloud = point_cloud[:, mask]
    mask = np.where(point_cloud[2] > -1.95)[0]
    point_cloud = point_cloud[:, mask]
    global_pcd = point_cloud

def get_cGAN_result():
    global global_img, global_nav
    img = Image.fromarray(cv2.cvtColor(global_img,cv2.COLOR_BGR2RGB))
    nav = Image.fromarray(cv2.cvtColor(global_nav,cv2.COLOR_BGR2RGB))
    img = img_trans(img)
    nav = img_trans(nav)
    input_img = torch.cat((img, nav), 0).unsqueeze(0).to(device)
    result = generator(input_img)
    result = result.cpu().data.numpy()[0][0]*127+128
    result = cv2.resize(result, (global_img.shape[1], global_img.shape[0]), interpolation=cv2.INTER_CUBIC)
    #print(result.max(), result.min())
    return result
    
def get_control(trajectory):
    global global_vel
    control = carla.VehicleControl()
    control.manual_gear_shift = True
    control.gear = 1
    control.throttle = 0.7

    x = trajectory['x']
    y = trajectory['y']
    vx = trajectory['vx']
    vy = trajectory['vy']
    yaw = np.arctan2(vy, vx)
    theta = np.arctan2(y, x)
    dist = np.sqrt(x**2+y**2)
    e = dist*np.sin(yaw-theta)
    K = 0.01
    Ks = 10.0
    tan_input = K*e/(Ks + global_vel)
    yaw_target = yaw + np.tan(tan_input)
    control.steer = np.clip(0.25*yaw_target, -1., 1.)
    return control

def main():
    global global_nav, global_vel
    client = carla.Client(config['host'], config['port'])
    client.set_timeout(config['timeout'])
    
    world = client.load_world('Town01')

    weather = carla.WeatherParameters(
        cloudiness=random.randint(0,10),
        precipitation=0,
        sun_altitude_angle=random.randint(50,90)
    )
    
    set_weather(world, weather)
    
    blueprint = world.get_blueprint_library()
    world_map = world.get_map()
    
    vehicle = add_vehicle(world, blueprint, vehicle_type='vehicle.audi.a2')
    # Enables or disables the simulation of physics on this actor.
    vehicle.set_simulate_physics(True)
    
    sensor_dict = {
        'camera':{
            'transform':carla.Transform(carla.Location(x=0.5, y=0.0, z=2.5)),
            'callback':image_callback,
            },
        'lidar':{
            'transform':carla.Transform(carla.Location(x=0.5, y=0.0, z=2.5)),
            'callback':lidar_callback,
            },
        }

    sm = SensorManager(world, blueprint, vehicle, sensor_dict)
    sm.init_all()
    time.sleep(0.3)
    
    spawn_points = world_map.get_spawn_points()
    waypoint_tuple_list = world_map.get_topology()
    origin_map = get_map(waypoint_tuple_list)

    agent = BasicAgent(vehicle, target_speed=MAX_SPEED)
    
    destination = get_random_destination(spawn_points)
    plan_map = replan(agent, destination, copy.deepcopy(origin_map))
    
    inverse_perspective_mapping = InversePerspectiveMapping(param, sensor_master)
    time.sleep(0.3)
    while True:
        if close2dest(vehicle, destination):
            destination = get_random_destination(spawn_points)
            plan_map = replan(agent, destination, copy.deepcopy(origin_map)) 
            
        #control = agent.run_step()
        #control.manual_gear_shift = False
        #vehicle.apply_control(control)
        #global_pos = vehicle.get_transform()
        v = vehicle.get_velocity()
        global_vel = np.sqrt(v.x**2+v.y**2+v.z**2)
        
        global_nav = get_nav(vehicle, plan_map)
        result = get_cGAN_result()
        
        img = copy.deepcopy(global_img)
        mask = np.where(result > 200)
        img[mask[0],mask[1]] = (255, 0, 0, 255)
        
        ipm_image = inverse_perspective_mapping.getIPM(result)
        cost_map = get_cost_map(ipm_image, global_pcd)
        #plan_cost_map = draw_traj(cost_map)
        
        plan_cost_map, trajectory = get_traj(cost_map, 0.7, show=True)
        control = get_control(trajectory)
        vehicle.apply_control(control)
        cv2.imshow('Result', img)
        cv2.imshow('Cost_map', plan_cost_map)
        cv2.waitKey(10)

    cv2.destroyAllWindows()
    sm.close_all()
    vehicle.destroy()
        
if __name__ == '__main__':
    main()