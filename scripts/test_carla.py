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

import numpy as np
from time import sleep

def image_callback(data):
    frame = data.frame_number
    #print(frame)
    #image.save_to_disk('output/%06d' % frame + '_raw.png')
      
def lidar_callback(data):
    lidar_data = np.frombuffer(data.raw_data, dtype=np.float32).reshape([-1, 3])
    #print(lidar_data.shape)
    
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

    try:
        while True:
            sleep(0.1)
            print('run')
    finally:
        sm.close_all()
        vehicle.destroy()
        
if __name__ == '__main__':
    main()