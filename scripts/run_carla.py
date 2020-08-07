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

from ff.collect_pm import CollectPerspectiveImage
from ff.carla_sensor import Sensor, CarlaSensorMaster

import os
import cv2
import time
import copy
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt

global_img = None
global_pcd = None

MAX_SPEED = 20
TRAJ_LENGTH = 25
vehicle_width = 2.2
longitudinal_sample_number_near = 8
longitudinal_sample_number_far = 0.5
lateral_sample_number = 20
lateral_step_factor = 1.0

parser = argparse.ArgumentParser(description='Params')
parser.add_argument('-d', '--data', type=int, default=1, help='data index')
parser.add_argument('-n', '--num', type=int, default=100000, help='total number')
args = parser.parse_args()

data_index = args.data
save_path = '/media/wang/DATASET/CARLA/'+str(data_index)+'/'

_sensor_dict = {
    'camera':{
        'transform':carla.Transform(carla.Location(x=1.5, y=0.0, z=2.0)),
        # 'callback':image_callback,
    },
    'lidar':{
        'transform':carla.Transform(carla.Location(x=1.5, y=0.0, z=2.0)),
        # 'callback':lidar_callback,
    },
}

class Param(object):
    def __init__(self):
        self.traj_length = float(TRAJ_LENGTH)
        self.target_speed = float(MAX_SPEED)
        self.vehicle_width = float(vehicle_width)
        self.longitudinal_sample_number_near = longitudinal_sample_number_near
        self.longitudinal_sample_number_far = longitudinal_sample_number_far
        self.lateral_sample_number = lateral_sample_number
        self.lateral_step_factor = lateral_step_factor
        
param = Param()
sensor = Sensor(_sensor_dict['camera']['transform'], config['camera'])
sensor_master = CarlaSensorMaster(sensor, _sensor_dict['camera']['transform'], binded=True)
collect_perspective = CollectPerspectiveImage(param, sensor_master)

def mkdir(path):
    if not os.path.exists(save_path+path):
        os.makedirs(save_path+path)

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
        
def main():
    global global_nav, global_control, global_pos, global_vel
    client = carla.Client(config['host'], config['port'])
    client.set_timeout(config['timeout'])
    
    #world = client.get_world()
    world = client.load_world('Town02')

    weather = carla.WeatherParameters(
        cloudiness=random.randint(0,10),
        precipitation=0,
        sun_altitude_angle=random.randint(70,90)
    )
    
    #world.set_weather(carla.WeatherParameters.ClearNoon)
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
    
    FPS = [str(time.time())]
    for cnt in range(args.num):
        if close2dest(vehicle, destination):
            destination = get_random_destination(spawn_points)
            plan_map = replan(agent, destination, copy.deepcopy(origin_map))
            
        if vehicle.is_at_traffic_light():
            traffic_light = vehicle.get_traffic_light()
            if traffic_light.get_state() == carla.TrafficLightState.Red:
                traffic_light.set_state(carla.TrafficLightState.Green)
                
        #control = agent.run_step()
        #control.manual_gear_shift = False
        #control = None
        #vehicle.apply_control(control)
        #nav = get_nav(vehicle, plan_map)
        global_pos = vehicle.get_transform()
        global_vel = vehicle.get_velocity()
        
        start_waypoint = agent._map.get_waypoint(agent._vehicle.get_location())
        end_waypoint = agent._map.get_waypoint(destination.location)
    
        route_trace = agent._trace_route(start_waypoint, end_waypoint)
        x = []
        y = []
        traj_pose_list = []
        for i in range(len(route_trace)-1):
            waypoint1 = route_trace[i][0].transform.location
            waypoint2 = route_trace[i+1][0].transform.location
            dist = waypoint1.distance(waypoint2)
            x.append(waypoint1.x)
            y.append(waypoint1.y)
            traj_pose_list.append((0, route_trace[i][0].transform))
        plt.plot([global_pos.location.x, x[0]], [global_pos.location.y, y[0]], color='r')
        plt.plot(x[:10], y[:10], color='b')
        plt.show()
        
        empty_image = collect_perspective.getPM(traj_pose_list, global_pos, global_img)
        break
        
        
        #cv2.imshow('Nav', nav)
        #cv2.imshow('Vision', global_img)
        #cv2.waitKey(10)
        index = str(time.time())

        if cnt % 100 == 0:
            FPS.append(index)
            print(round(1/((float(FPS[-1])-float(FPS[-2]))/100),1))
        
    cv2.destroyAllWindows()
    vehicle.destroy()
    sm.close_all()
        
if __name__ == '__main__':
    main()