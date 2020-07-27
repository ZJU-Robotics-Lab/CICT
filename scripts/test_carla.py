#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from os.path import join, dirname
sys.path.insert(0, join(dirname(__file__), '..'))

import simulator
simulator.load('/home/wang/CARLA_0.9.9.4')
import carla
from simulator import config, set_weather, add_vehicle
from simulator.sensor_manager import SensorManager

import cv2
import time
import random
import argparse
import numpy as np
from PIL import Image
from datetime import datetime

import torch
import torchvision.transforms as transforms
from learning.models import GeneratorUNet

from utils.local_planner_sim import get_cost_map, get_cmd
from utils.camera_info_sim import camera2lidar

from device.lidar import Visualizer

class PDVisualizer(Visualizer):        
    def update(self):
        global global_pc
        self.points = global_pc.T
        self.set_plotdata(
            points=self.points,
            color=(0.,1.,1.,1.)
        )
        
parser = argparse.ArgumentParser()
parser.add_argument('--img_height', type=int, default=128, help='size of image height')
parser.add_argument('--img_width', type=int, default=256, help='size of image width')
opt = parser.parse_args()

random.seed(datetime.now())
torch.manual_seed(999)
torch.cuda.manual_seed(999)
torch.set_num_threads(16)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
generator = GeneratorUNet()

generator = generator.to(device)
generator.load_state_dict(torch.load('../ckpt/g.pth', map_location=device))
generator.eval()

img_trans_ = [
    transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
img_trans = transforms.Compose(img_trans_)

global_img = None
global_pc = None
global_pc2 = None

def get_img():
    img = global_img#cv2.imread("img.png")
    img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

    nav = cv2.imread("../utils/nav.png")
    nav = Image.fromarray(cv2.cvtColor(nav,cv2.COLOR_BGR2RGB))
    img = img_trans(img)
    nav = img_trans(nav)
    input_img = torch.cat((img, nav), 0).unsqueeze(0)
    return input_img.to(device)
    

def get_net_result(input_img):
    with torch.no_grad():
        input_img = input_img#.to(device)
        result = generator(input_img)
        return result
    
def inverse_perspective_mapping(img):
    global global_pc
    point_cloud = global_pc
    img = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_AREA)
    res = np.where(img > 100)
    image_uv = np.stack([res[1],res[0]])
    trans_pc = camera2lidar(image_uv)
    img = get_cost_map(trans_pc, point_cloud, False)
    yaw = get_cmd(img, show=False)
    #print(2.4*yaw)
    return img, yaw
    
def image_callback(data):
    global global_img
    array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8")) 
    array = np.reshape(array, (data.height, data.width, 4)) # RGBA format
    array = array[:, :, :3] # Take only RGB
    array = array[:, :, ::-1] # BGR
    global_img = array
    
def lidar_callback(data):
    global global_pc
    lidar_data = np.frombuffer(data.raw_data, dtype=np.float32).reshape([-1, 3])
    point_cloud = np.stack([-lidar_data[:,1], -lidar_data[:,0], -lidar_data[:,2]])
    mask = np.where((point_cloud[0] > 1.0)|(point_cloud[0] < -4.0)|(point_cloud[1] > 1.2)|(point_cloud[1] < -1.2))[0]
    point_cloud = point_cloud[:, mask]
    global_pc = point_cloud
    
def imu_callback(data):
    # Measures linear acceleration in m/s^2
    data.accelerometer
    # Measures angular velocity in rad/sec
    data.gyroscope
    # Orientation in radians. North is (0.0, -1.0, 0.0) in UE
    data.compass

def gnss_callback(data):
    data.latitude
    data.longitude
    data.altitude

def main():
    client = carla.Client(config['host'], config['port'])
    client.set_timeout(config['timeout'])
    
    world = client.get_world()
    weather = carla.WeatherParameters(
        cloudiness=30.0,
        precipitation=30.0,
        sun_altitude_angle=50.0
    )
    set_weather(world, weather)
    
    blueprint = world.get_blueprint_library()
    world_map = world.get_map()
    
    vehicle = add_vehicle(world, blueprint, vehicle_type='vehicle.bmw.grandtourer')
    # Enables or disables the simulation of physics on this actor.
    vehicle.set_simulate_physics(True)

    sensor_dict = {
        'camera':{
            'transform':carla.Transform(carla.Location(x=1.5, y=0.0, z=2.0)),
            'callback':image_callback,
            },
        'lidar':{
            'transform':carla.Transform(carla.Location(x=1.5, y=0.0, z=2.0)),
            'callback':lidar_callback,
            },
        'gnss':{
            'transform':carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0)),
            'callback':gnss_callback,
            },
        'imu':{
            'transform':carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0)),
            'callback':imu_callback,
            },
        }

    sm = SensorManager(world, blueprint, vehicle, sensor_dict)
    sm.init_all()
    time.sleep(0.3)
    #v = PDVisualizer(None)

    try:
        while True:
            input_img = get_img()
            result = get_net_result(input_img)[0][0]
            result = result.cpu().data.numpy()*255+255
            costmap, yaw = inverse_perspective_mapping(result)
            cv2.imshow('Result', costmap)
            #cv2.imshow('Result', result)
            cv2.imshow('Raw image', global_img)
            vehicle.apply_control(carla.VehicleControl(
                throttle=0.3, 
                steer = yaw, 
                reverse=False,
                manual_gear_shift=True,
                gear=1,
                brake=0.0,
                hand_brake=False
            ))
            
            #vehicle.get_angular_velocity().z
            #vehicle.get_velocity().x

            cv2.waitKey(1)
            #v.animation()
            #v.close()
            #break
            
        cv2.destroyAllWindows()
    finally:
        sm.close_all()
        vehicle.destroy()
        
if __name__ == '__main__':
    main()