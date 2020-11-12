#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from os.path import join, dirname
sys.path.insert(0, join(dirname(__file__), '../../'))
sys.path.insert(0, join(dirname(__file__), '../'))

import simulator
simulator.load('/home/wang/CARLA_0.9.9.4')
import carla
sys.path.append('/home/wang/CARLA_0.9.9.4/PythonAPI/carla')
from agents.navigation.basic_agent import BasicAgent

from simulator import config, set_weather, add_vehicle
from simulator.sensor_manager import SensorManager
from utils.navigator_sim import get_map, get_nav, replan, close2dest
from learning.models import GeneratorUNet
from learning.path_model import Model_COS, ModelGRU2
from utils import fig2data, add_alpha_channel

from ff_collect_pm_data import sensor_dict
from ff.collect_ipm import InversePerspectiveMapping
from ff.carla_sensor import Sensor, CarlaSensorMaster

import os
import cv2
import time
import copy
import threading
import random
import argparse
import numpy as np
from PIL import Image, ImageDraw
from datetime import datetime
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})

import torch
from torch.autograd import grad
import torchvision.transforms as transforms

global_img = None
global_pcd = None
global_nav = None
global_v0 = 0.
global_vel = 0.
global_plan_time = 0.
global_trajectory = None
start_control = False
global_vehicle = None
global_plan_map = None
global_cost_map = None
global_transform = None
max_steer_angle = 0.
draw_cost_map = None
global_ipm_image = np.zeros((200,400), dtype=np.uint8)
global_ipm_image.fill(255)

global_trans_costmap_list = []


MAX_SPEED = 30
img_height = 128
img_width = 256
longitudinal_length = 25.0 # [m]

random.seed(datetime.now())
torch.manual_seed(999)
torch.cuda.manual_seed(999)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

generator = GeneratorUNet()
generator = generator.to(device)
generator.load_state_dict(torch.load('../../ckpt/sim-obs/g.pth'))
model = ModelGRU2().to(device)
model.load_state_dict(torch.load('../../ckpt/gru/model.pth'))
generator.eval()
model.eval()

parser = argparse.ArgumentParser(description='Params')
parser.add_argument('-d', '--data', type=int, default=1, help='data index')
parser.add_argument('-n', '--num', type=int, default=100000, help='total number')
parser.add_argument('--width', type=int, default=400, help='image width')
parser.add_argument('--height', type=int, default=200, help='image height')
parser.add_argument('--max_dist', type=float, default=25., help='max distance')
parser.add_argument('--max_t', type=float, default=3., help='max time')
parser.add_argument('--scale', type=float, default=25., help='longitudinal length')
parser.add_argument('--dt', type=float, default=0.05, help='discretization minimum time interval')
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
    # for resnet backbone
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
cost_map_trans = transforms.Compose(cost_map_transforms_)
        
class Param(object):
    def __init__(self):
        self.longitudinal_length = longitudinal_length
        self.ksize = 21
        
param = Param()
sensor = Sensor(sensor_dict['camera']['transform'], config['camera'])
sensor_master = CarlaSensorMaster(sensor, sensor_dict['camera']['transform'], binded=True)
inverse_perspective_mapping = InversePerspectiveMapping(param, sensor_master)

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
    
    kernel_size = (3, 3)
    img = cv2.dilate(img,kernel_size,iterations = 3)
    
    img = cv2.addWeighted(img,0.5,img2,0.5,0)
    
    mask = np.where((img2 < 50))
    u = mask[0]
    v = mask[1]
    img[u, v] = 0

    return img

def xy2uv(x, y):
    pixs_per_meter = args.height/args.scale
    u = (args.height-x*pixs_per_meter).astype(int)
    v = (y*pixs_per_meter+args.width//2).astype(int)
    return u, v

def mkdir(path):
    os.makedirs(save_path+path, exist_ok=True)

def image_callback(data):
    global global_img, global_plan_time, global_vehicle, global_plan_map,global_nav, global_transform, global_v0
    global_plan_time = time.time()
    array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8")) 
    array = np.reshape(array, (data.height, data.width, 4)) # RGBA format
    global_img = array
    global_transform = global_vehicle.get_transform()
    try:
        global_nav = get_nav(global_vehicle, global_plan_map)
        v = global_vehicle.get_velocity()
        global_v0 = np.sqrt(v.x**2+v.y**2+v.z**2)
    except:
        pass
    
def lidar_callback(data):
    global global_pcd
    lidar_data = np.frombuffer(data.raw_data, dtype=np.float32).reshape([-1, 3])
    point_cloud = np.stack([-lidar_data[:,1], -lidar_data[:,0], -lidar_data[:,2]])
    mask = np.where(point_cloud[2] > -2.3)[0]
    point_cloud = point_cloud[:, mask]
    global_pcd = point_cloud
    generate_costmap()
    print(time.time())

def get_cGAN_result(img, nav):
    img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    nav = Image.fromarray(cv2.cvtColor(nav,cv2.COLOR_BGR2RGB))
    img = img_trans(img)
    nav = img_trans(nav)
    input_img = torch.cat((img, nav), 0).unsqueeze(0).to(device)
    result = generator(input_img)
    result = result.cpu().data.numpy()[0][0]*127+128
    result = cv2.resize(result, (global_img.shape[1], global_img.shape[0]), interpolation=cv2.INTER_CUBIC)
    #print(result.max(), result.min())
    return result

cnt = 0
def visualize(img, costmap, nav, curve=None):
    global global_vel, cnt
    #costmap = cv2.cvtColor(costmap,cv2.COLOR_GRAY2RGB)
    text = "speed: "+str(round(3.6*global_vel, 1))+' km/h'
    cv2.putText(img, text, (20, 30), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255), 2)
    
    nav = cv2.resize(nav, (240, 200), interpolation=cv2.INTER_CUBIC)
    down_img = np.hstack([costmap, nav])
    down_img = add_alpha_channel(down_img)
    show_img = np.vstack([img, down_img])
    #print(show_img.shape, curve.shape)
    if curve is not None:
        curve = cv2.cvtColor(curve,cv2.COLOR_BGRA2RGBA)
        left_img = cv2.resize(curve, (int(curve.shape[1]*show_img.shape[0]/curve.shape[0]), show_img.shape[0]), interpolation=cv2.INTER_CUBIC)
        show_img = np.hstack([show_img, left_img])
    cv2.imshow('Visualization', show_img)
    #cv2.imwrite('result/images/gru01/'+str(cnt)+'.png', show_img)
    cv2.waitKey(10)
    cnt += 1

def draw_traj(cost_map, output):
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
    return cost_map
        
def get_traj(plan_time):
    global global_v0, draw_cost_map, global_trans_costmap_list

    t = torch.arange(0, 0.99, args.dt).unsqueeze(1).to(device)
    t.requires_grad = True
    
    trans_img = torch.stack(global_trans_costmap_list)
    img = trans_img.expand(len(t),10,1,args.height, args.width)
    # for resnet backbone
    #img = trans_img.expand(len(t),3,args.height, args.width)
    img = img.to(device)
    img.requires_grad = True
    v_0 = torch.FloatTensor([global_v0]).expand(len(t),1)
    v_0 = v_0.to(device)

    output = model(img, t, v_0)
    vx = grad(output[:,0].sum(), t, create_graph=True)[0][:,0]*(args.max_dist/args.max_t)
    vy = grad(output[:,1].sum(), t, create_graph=True)[0][:,0]*(args.max_dist/args.max_t)
    
    ax = grad(vx.sum(), t, create_graph=True)[0][:,0]/args.max_t
    ay = grad(vy.sum(), t, create_graph=True)[0][:,0]/args.max_t

    output_axy = torch.cat([ax.unsqueeze(1), ay.unsqueeze(1)], dim=1)

    x = output[:,0]*args.max_dist
    y = output[:,1]*args.max_dist
    
    theta_a = torch.atan2(ay, ax)
    theta_v = torch.atan2(vy, vx)
    sign = torch.sign(torch.cos(theta_a-theta_v))
    a = torch.mul(torch.norm(output_axy, dim=1), sign.flatten()).unsqueeze(1)
    
    # draw
    draw_cost_map = draw_traj(global_cost_map, output)
    
    vx = vx.data.cpu().numpy()
    vy = vy.data.cpu().numpy()
    x = x.data.cpu().numpy()
    y = y.data.cpu().numpy()
    ax = ax.data.cpu().numpy()
    ay = ay.data.cpu().numpy()
    a = a.data.cpu().numpy()
    trajectory = {'time':plan_time, 'x':x, 'y':y, 'vx':vx, 'vy':vy, 'ax':ax, 'ay':ay, 'a':a}
    return trajectory

def generate_costmap():
    global global_pcd, global_ipm_image, global_cost_map, global_trans_costmap_list
    cost_map = get_cost_map(global_ipm_image, global_pcd)
    global_cost_map = cost_map
    img = Image.fromarray(cv2.cvtColor(cost_map,cv2.COLOR_BGR2RGB)).convert('L')
    trans_img = cost_map_trans(img)
    while len(global_trans_costmap_list) < 10:
        global_trans_costmap_list.append(trans_img)
    global_trans_costmap_list.pop(0)
    global_trans_costmap_list.append(trans_img)

def make_plan():
    global global_img, global_nav, global_pcd, global_plan_time, global_trajectory,start_control, global_ipm_image
    while True:
        t1 = time.time()
        plan_time = global_plan_time
        # 1. get cGAN result
        result = get_cGAN_result(global_img, global_nav)
        # 2. inverse perspective mapping and get costmap
        img = copy.deepcopy(global_img)
        mask = np.where(result > 200)
        img[mask[0],mask[1]] = (255, 0, 0, 255)
        
        ipm_image = inverse_perspective_mapping.getIPM(result)
        global_ipm_image = ipm_image

        # 3. get trajectory
        global_trajectory = get_traj(plan_time)
        #time.sleep(0.2)

        if not start_control:
            start_control = True
        t2 = time.time()
        print('time:', 1000*(t2-t1))
            

def get_transform(transform, org_transform):
    x = transform.location.x
    y = transform.location.y
    yaw = transform.rotation.yaw
    x0 = org_transform.location.x
    y0 = org_transform.location.y
    yaw0 = org_transform.rotation.yaw
    
    dx = x - x0
    dy = y - y0
    dyaw = yaw - yaw0
    return dx, dy, dyaw
    
def get_control(x, y, vx, vy, ax, ay):
    global global_vel, max_steer_angle, global_a
    Kx = 0.3
    Kv = 3.0*1.5
    
    Ky = 2.5e-2
    K_theta = 0.05
    
    control = carla.VehicleControl()
    control.manual_gear_shift = True
    control.gear = 1

    v_r = np.sqrt(vx**2+vy**2)

    yaw = np.arctan2(vy, vx)
    theta_e = yaw
    
    #k = (vx*ay-vy*ax)/(v_r**3)
    w_r = (vx*ay-vy*ax)/(v_r**2)
    theta = np.arctan2(y, x)
    dist = np.sqrt(x**2+y**2)
    y_e = dist*np.sin(yaw-theta)
    x_e = dist*np.cos(yaw-theta)
    v_e = v_r - global_vel
    ####################################
    
    #v = v_r*np.cos(theta_e) + Kx*x_eglobal global_trajectory
    w = w_r + v_r*(Ky*y_e + K_theta*np.sin(theta_e))
    
    steer_angle = np.arctan(w*2.405/global_vel)
    steer = steer_angle/max_steer_angle
    #####################################
    
    #throttle = Kx*x_e + Kv*v_e+0.7
    #throttle = 0.7 +(Kx*x_e + Kv*v_e)*0.06
    #throttle = Kx*x_e + Kv*v_e+0.5
    throttle = Kx*x_e + Kv*v_e + global_a
    
    control.brake = 0.0
    if throttle > 0:
        control.throttle = np.clip(throttle, 0., 1.)
    else:
        #control.brake = np.clip(-0.05*throttle, 0., 1.)
        control.brake = np.clip(100*throttle, 0., 1.)
    control.steer = np.clip(steer, -1., 1.)
    return control

#def show_traj(trajectory, show=False):
def show_traj():
    global global_trajectory
    max_x = 30.
    max_y = 30.
    max_speed = 12.0
    while True:
        trajectory = global_trajectory
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        x = trajectory['x']
        y = trajectory['y']
        ax1.plot(x, y, label='trajectory', color = 'b', linewidth=3)
        ax1.set_xlabel('Tangential/(m)')
        ax1.set_ylabel('Normal/(m)')
        ax1.set_xlim([0., max_x])
        ax1.set_ylim([-max_y, max_y])
        plt.legend(loc='lower right')
        
        t = x[-1]*np.arange(0, 1.0, 1./x.shape[0])
        a = trajectory['a']
        vx = trajectory['vx']
        vy = trajectory['vy']
        v = np.sqrt(np.power(vx, 2), np.power(vy, 2))
        angle = np.arctan2(vy, vx)/np.pi*max_speed
        ax2 = ax1.twinx()
        ax2.plot(t, v, label='speed', color = 'r', linewidth=2)
        ax2.plot(t, a, label='acc', color = 'y', linewidth=2)
        ax2.plot(t, angle, label='angle', color = 'g', linewidth=2)
        ax2.set_ylabel('Velocity/(m/s)')
        ax2.set_ylim([-max_speed, max_speed])
        plt.legend(loc='upper right')
        plt.show()
    
    
def main():
    global global_nav, global_vel, start_control, global_plan_map, global_vehicle, global_cost_map, global_transform, max_steer_angle, global_a, draw_cost_map
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
    #vehicle = add_vehicle(world, blueprint, vehicle_type='vehicle.yamaha.yzf')
    #vehicle = add_vehicle(world, blueprint, vehicle_type='vehicle.*.*')
    global_vehicle = vehicle
    # Enables or disables the simulation of physics on this actor.
    vehicle.set_simulate_physics(True)
    physics_control = vehicle.get_physics_control()
    max_steer_angle = np.deg2rad(physics_control.wheels[0].max_steer_angle)
    
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
    
    #spawn_points = world_map.get_spawn_points()
    waypoint_tuple_list = world_map.get_topology()
    origin_map = get_map(waypoint_tuple_list)

    agent = BasicAgent(vehicle, target_speed=MAX_SPEED)
    
    # prepare map
    destination = carla.Transform()
    destination.location = world.get_random_location_from_navigation()
    #destination = get_random_destination(spawn_points)
    global_plan_map = replan(agent, destination, copy.deepcopy(origin_map))

    # start to plan
    plan_thread = threading.Thread(target = make_plan, args=())
    visualization_thread = threading.Thread(target = show_traj, args=())

    while True:
        if (global_img is not None) and (global_nav is not None) and (global_pcd is not None):
            plan_thread.start()
            break
        else:
            time.sleep(0.001)
    
    # wait for the first plan result
    while not start_control:
        time.sleep(0.001)
    
    visualization_thread.start()
    # start to control
    print('Start to control')
    avg_dt = 1.0
    distance = 0.
    last_location = vehicle.get_location()
    while True:
        # change destination
        if close2dest(vehicle, destination):
            #destination = get_random_destination(spawn_points)
            destination = carla.Transform()
            destination.location = world.get_random_location_from_navigation()
            global_plan_map = replan(agent, destination, copy.deepcopy(origin_map)) 

        v = global_vehicle.get_velocity()
        a = global_vehicle.get_acceleration()
        global_vel = np.sqrt(v.x**2+v.y**2+v.z**2)
        global_a = np.sqrt(a.x**2+a.y**2+a.z**2)
        #steer_angle = global_vehicle.get_control().steer*max_steer_angle
        #w = global_vel*np.tan(steer_angle)/2.405
        
        control_time = time.time()
        dt = control_time - global_trajectory['time']
        avg_dt = 0.99*avg_dt + 0.01*dt
        #print(round(avg_dt, 3))
        index = int((dt/args.max_t)//args.dt)
        if index > 0.99/args.dt:
            continue
        
        location = vehicle.get_location()
        distance += location.distance(last_location)
        last_location = location
        #print(round(distance, 1))
        
        transform = vehicle.get_transform()
        dx, dy, dyaw = get_transform(transform, global_transform)
        dyaw = -dyaw
        
        _x = global_trajectory['x'][index] - dx
        _y = global_trajectory['y'][index] - dy
        x = _x*np.cos(dyaw) + _y*np.sin(dyaw)
        y = _y*np.cos(dyaw) - _x*np.sin(dyaw)
        
        _vx = global_trajectory['vx'][index]
        _vy = global_trajectory['vy'][index]
        vx = _vx*np.cos(dyaw) + _vy*np.sin(dyaw)
        vy = _vy*np.cos(dyaw) - _vx*np.sin(dyaw)
        
        _ax = global_trajectory['ax'][index]
        _ay = global_trajectory['ay'][index]
        ax = _ax*np.cos(dyaw) + _ay*np.sin(dyaw)
        ay = _ay*np.cos(dyaw) - _ax*np.sin(dyaw)
        
        control = get_control(x, y, vx, vy, ax, ay)
        vehicle.apply_control(control)
        
        #print(global_vel*np.tan(control.steer)/w)
        curve = None#show_traj()
        visualize(global_img, draw_cost_map, global_nav, curve)
        
        #time.sleep(1/60.)

    cv2.destroyAllWindows()
    sm.close_all()
    vehicle.destroy()
        
if __name__ == '__main__':
    main()