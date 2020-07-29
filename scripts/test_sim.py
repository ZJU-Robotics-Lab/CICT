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
import numpy as np

import socket
UDP_IP = "127.0.0.1"
UDP_PORT = 6666

sock = socket.socket(socket.AF_INET, # Internet
             socket.SOCK_DGRAM) # UDP

global_img = None
    
def image_callback(data):
    global global_img
    array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8")) 
    array = np.reshape(array, (data.height, data.width, 4)) # RGBA format
    array = array[:, :, :3] # Take only RGB
    array = array[:, :, ::-1] # BGR
    global_img = array
    
def lidar_callback(data):
    lidar_data = np.frombuffer(data.raw_data, dtype=np.float32).reshape([-1, 3])
    point_cloud = np.stack([-lidar_data[:,1], -lidar_data[:,0], -lidar_data[:,2]])
    mask = np.where((point_cloud[0] > 1.0)|(point_cloud[0] < -4.0)|(point_cloud[1] > 1.2)|(point_cloud[1] < -1.2))[0]
    point_cloud = point_cloud[:, mask]
    mask = np.where(point_cloud[2] > -1.95)[0]
    point_cloud = point_cloud[:, mask]
    if point_cloud.shape[1] < 5400:
        sock.sendto(point_cloud.T.tobytes(), (UDP_IP, UDP_PORT))
    else:
        sock.sendto(point_cloud[:,:5400].T.tobytes(), (UDP_IP, UDP_PORT))
    
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

    try:
        while True:
            cv2.imshow('Raw image', global_img)
            vehicle.apply_control(carla.VehicleControl(
                throttle=0.3, 
                steer = 0.0, 
                reverse=False,
                manual_gear_shift=True,
                gear=1,
                brake=0.0,
                hand_brake=False
            ))
            
            #vehicle.get_angular_velocity().z
            #vehicle.get_velocity().x

            cv2.waitKey(1)
            #break
            
        cv2.destroyAllWindows()
    finally:
        sm.close_all()
        vehicle.destroy()
        
if __name__ == '__main__':
    main()