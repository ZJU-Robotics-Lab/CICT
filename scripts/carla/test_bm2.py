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
from utils import fig2data, add_alpha_channel

from ff_collect_pm_data import sensor_dict
from ff.collect_ipm import InversePerspectiveMapping
from ff.carla_sensor import Sensor, CarlaSensorMaster
from ff.capac_controller import CapacController
import carla_utils as cu

import os
import cv2
import time
import copy
import random
import argparse
import numpy as np
from PIL import Image, ImageDraw
from datetime import datetime
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})

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
global_view_img = None
state0 = None
global_collision = False
last_collision_time = time.time()
global_ipm_image = np.zeros((200,400), dtype=np.uint8)
global_ipm_image.fill(255)

global_trans_costmap_list = []
global_trans_costmap_dict = {}


MAX_SPEED = 30
img_height = 128
img_width = 256
longitudinal_length = 25.0 # [m]

parser = argparse.ArgumentParser(description='Params')
parser.add_argument('-d', '--data', type=int, default=1, help='data index')
parser.add_argument('-t', '--town', type=int, default=1, help='twon index')
parser.add_argument('-s', '--save', type=bool, default=False, help='save result')
parser.add_argument('--width', type=int, default=400, help='image width')
parser.add_argument('--height', type=int, default=200, help='image height')
parser.add_argument('--max_dist', type=float, default=25., help='max distance')
parser.add_argument('--max_t', type=float, default=3., help='max time')
parser.add_argument('--scale', type=float, default=25., help='longitudinal length')
parser.add_argument('--dt', type=float, default=0.03, help='discretization minimum time interval')
parser.add_argument('--rnn_steps', type=int, default=10, help='rnn readout steps')
args = parser.parse_args()

data_index = args.data
save_path = '/media/wang/DATASET/CARLA/town01/'+str(data_index)+'/'
        
class Param(object):
    def __init__(self):
        self.longitudinal_length = longitudinal_length
        self.ksize = 21
        
param = Param()
sensor = Sensor(sensor_dict['camera']['transform'], config['camera'])
sensor_master = CarlaSensorMaster(sensor, sensor_dict['camera']['transform'], binded=True)
inverse_perspective_mapping = InversePerspectiveMapping(param, sensor_master)

def read(file_path, town_map):
    poses = []
    file = open(file_path)
    while True:
        line = file.readline()
        if not line:
            break
        if line == '\n':
            continue

        line_list = line.split(',')
        start_x, start_y = eval(line_list[0]), eval(line_list[1])
        end_x, end_y = eval(line_list[2]), eval(line_list[3])

        start_transform = town_map.get_waypoint(carla.Location(x=start_x, y=start_y)).transform
        end_transform = town_map.get_waypoint(carla.Location(x=end_x, y=end_y)).transform
        poses.append([start_transform, end_transform])
    file.close()
    return poses

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
    
def view_image_callback(data):
    global global_view_img
    array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8")) 
    array = np.reshape(array, (data.height, data.width, 4)) # RGBA format
    global_view_img = array


def collision_callback(data):
    global global_collision
    if global_collision: return
    global_collision = True
    print(data)


cnt = 0
def visualize(img, costmap, nav, curve=None):
    global global_vel, cnt
    #if cnt % 3 == 0: cv2.imwrite('result/images/result4/'+str(cnt)+'.png', img)
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
    if args.save: cv2.imwrite('result/images/img02/'+str(cnt)+'.png', show_img)
    cv2.waitKey(5)
    cnt += 1


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
        try:
            ipm_image = inverse_perspective_mapping.getIPM(result)
            global_ipm_image = ipm_image
        except:
            pass

        # 3. get trajectory
        #time.sleep(1.2-0.11)
        try:
            global_trajectory = get_traj(plan_time)
        except:
            pass

        if not start_control:
            start_control = True
        t2 = time.time()
        #print('time:', 1000*(t2-t1))
            

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
    
    Ky = 5.0e-3
    K_theta = 0.10
    
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
    
    steer_angle = np.arctan(w*2.405/global_vel) if abs(global_vel) > 0.001 else 0.
    steer = steer_angle/max_steer_angle
    #####################################
    
    #throttle = Kx*x_e + Kv*v_e+0.7
    #throttle = 0.7 +(Kx*x_e + Kv*v_e)*0.06
    #throttle = Kx*x_e + Kv*v_e+0.5
    throttle = Kx*x_e + Kv*v_e + global_a
    # MAGIC !
    #if throttle > 0 and abs(global_vel) < 0.8 and abs(v_r) < 1.0:
    if throttle > 0 and abs(global_vel) < 0.8 and abs(v_r) < 1.2:
        throttle = -1
    
    control.brake = 0.0
    if throttle > 0:
        control.throttle = np.clip(throttle, 0., 1.)
    else:
        #control.brake = np.clip(-0.05*throttle, 0., 1.)
        control.brake = np.clip(abs(100*throttle), 0., 1.)
    control.steer = np.clip(steer, -1., 1.)
    return control

def show_traj(save=False):
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
        
        t = max_x*np.arange(0, 1.0, 1./x.shape[0])
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
        if not save:
            plt.show()
        else:
            image = fig2data(fig)
            plt.close('all')
            return image
    
def main():
    global global_nav, global_vel, start_control, global_plan_map, global_vehicle, global_cost_map, global_transform, max_steer_angle, global_a, draw_cost_map, state0, global_collision, last_collision_time
    client = carla.Client(config['host'], config['port'])
    client.set_timeout(config['timeout'])
    
    town = 'Town0'+str(args.town)
    world = client.load_world(town)
    world_map = world.get_map()
    poses = read(town+'/navigation_with_dynamic_obstacles.csv', world_map)
    """
    weather = carla.WeatherParameters(
        cloudiness=random.randint(0,10),
        precipitation=0,
        sun_altitude_angle=random.randint(50,90)
    )
    
    set_weather(world, weather)
    """
    #world.set_weather(carla.WeatherParameters.ClearNoon)
    # [1, 3, 6, 8] for training
    train_weather = random.sample([carla.WeatherParameters.ClearNoon, carla.WeatherParameters.WetNoon, carla.WeatherParameters.HardRainNoon, carla.WeatherParameters.ClearSunset], 1)[0]
    test_weather = random.sample([carla.WeatherParameters.WetSunset,carla.WeatherParameters.SoftRainSunset], 1)[0]
    #world.set_weather(carla.WeatherParameters.HardRainSunset)
    weather = train_weather if args.town == 1 else test_weather
    world.set_weather(weather)
    
    blueprint = world.get_blueprint_library()
    
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
        'camera:view':{
            #'transform':carla.Transform(carla.Location(x=0.0, y=4.0, z=4.0), carla.Rotation(pitch=-30, yaw=-60)),
            'transform':carla.Transform(carla.Location(x=-3.0, y=0.0, z=6.0), carla.Rotation(pitch=-45)),
            #'transform':carla.Transform(carla.Location(x=4.0, y=0.0, z=10.0), carla.Rotation(pitch=-90)),
            'callback':view_image_callback,
            },
        'collision':{
            'transform':carla.Transform(carla.Location(x=0.5, y=0.0, z=2.5)),
            'callback':collision_callback,
            },
        }

    sm = SensorManager(world, blueprint, vehicle, sensor_dict)
    sm.init_all()
    
    #spawn_points = world_map.get_spawn_points()
    waypoint_tuple_list = world_map.get_topology()
    origin_map = get_map(waypoint_tuple_list)

    # prepare map
    #destination = carla.Transform()
    #destination.location = world.get_random_location_from_navigation()
    #global_plan_map = replan(agent, destination, copy.deepcopy(origin_map))
    
    agent = BasicAgent(vehicle, target_speed=30)
    total_eps = 1#13#21
    success_eps = total_eps-1#0
    task = random.sample(poses, 1)[0]
    #task = poses[total_eps-1]#poses[0]
    vehicle.set_transform(task[0])
    destination = task[1]
    global_plan_map = replan(agent, destination, copy.deepcopy(origin_map))

    while True:
        if (global_img is not None) and (global_nav is not None):
            #plan_thread.start()
            break
        else:
            time.sleep(0.001)
    
    # start to control
    print('Start to control')
    
    #ctrller = CapacController(world, vehicle, 30)
    agent.set_destination([destination.location.x, destination.location.y, destination.location.z])
    while True:
        # change destination
        if close2dest(vehicle, destination, 8) or global_collision:
            train_weather = random.sample([carla.WeatherParameters.ClearNoon, carla.WeatherParameters.WetNoon, carla.WeatherParameters.HardRainNoon, carla.WeatherParameters.ClearSunset], 1)[0]
            test_weather = random.sample([carla.WeatherParameters.WetSunset,carla.WeatherParameters.SoftRainSunset], 1)[0]
            weather = train_weather if args.town == 1 else test_weather
            world.set_weather(weather)
            
    
            if global_collision:
                collision_time = time.time()
                if collision_time - last_collision_time < 2:
                    total_eps -= 1
                last_collision_time = collision_time
                global_collision = False
                cv2.imwrite('result/collision2/'+str(time.time())+'.png', global_view_img)
            else:
                success_eps += 1
            
            print(time.time(), 'ID:',total_eps, round(100*success_eps/total_eps, 2), '%  ', success_eps,'/',total_eps)
            total_eps += 1
            
            #task = random.sample(poses, 1)[0]
            if total_eps >= len(poses):
                break
            #task = poses[total_eps]
            while True:
                task = random.sample(poses, 1)[0]
                #a = carla.Waypoint(transform=task[0])
                #print(task[0], type(task[0]), type(carla.Waypoint(transform=task[0])))
                route = agent._trace_route(world_map.get_waypoint(task[0].location), world_map.get_waypoint(task[1].location))
                #print(route[0].transform)
                
                #vehicle.set_transform(task[0])
                vehicle.set_transform(route[0][0].transform)
                time.sleep(0.1)
                if abs(vehicle.get_location().z) > 3:
                    continue

                destination = task[1]
                global_plan_map = replan(agent, destination, copy.deepcopy(origin_map))
                break
            global_collision = False
            last_collision_time = time.time()
            #agent.set_destination([destination.location.x, destination.location.y, destination.location.z])
                

        v = global_vehicle.get_velocity()
        a = global_vehicle.get_acceleration()
        global_vel = np.sqrt(v.x**2+v.y**2+v.z**2)
        global_a = np.sqrt(a.x**2+a.y**2+a.z**2)
        
        control = agent.run_step()
        control.manual_gear_shift = False
        time.sleep(0.05)
        vehicle.apply_control(control)
        
        
        curve = None#show_traj(True)
        #cv2.imshow('Visualization', global_view_img)
        #cv2.waitKey(5)
        draw_cost_map = Image.new('RGB', (400, 200), (255, 255, 255))
        visualize(global_view_img, draw_cost_map, global_nav, curve)
        #time.sleep(1/60.)

    cv2.destroyAllWindows()
    sm.close_all()
    vehicle.destroy()
        
if __name__ == '__main__':
    main()