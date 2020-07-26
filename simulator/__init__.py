#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import glob
import random

config = {
    'host': 'localhost',
    'port': 2000,
    'timeout': 5.0,
    'camera':{
        'img_length': 800,
        'img_width': 600,
        'fov': 90,
        'fps': 30,
        },
    'lidar':{
        'rpm': 10,
        },
    'imu':{
        'fps': 400,
        },
    'gnss':{
        'fps': 20,
        },
}

def load(path='/home/wang/CARLA_0.9.9.4'):
    try:
        sys.path.append(path+'/PythonAPI')
        sys.path.append(glob.glob(path+'/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
            sys.version_info.major,
            sys.version_info.minor,
            'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    except:
        print('Fail to load carla library')   

def add_vehicle(world, blueprint, vehicle_type='vehicle.bmw.grandtourer'):
    bp = random.choice(blueprint.filter(vehicle_type))
    if bp.has_attribute('color'):
        color = random.choice(bp.get_attribute('color').recommended_values)
        bp.set_attribute('color', color)
    transform = random.choice(world.get_map().get_spawn_points())
    vehicle = world.spawn_actor(bp, transform)
    return vehicle

# https://carla.readthedocs.io/en/latest/ref_sensors/
def add_camera(world, blueprint, vehicle, transform):
    camera_bp = blueprint.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', str(config['camera']['img_length']))
    camera_bp.set_attribute('image_size_y', str(config['camera']['img_width']))
    camera_bp.set_attribute('fov', str(config['camera']['fov']))
    camera_bp.set_attribute('sensor_tick', str(1./config['camera']['fps']))
    camera = world.spawn_actor(camera_bp, transform, attach_to=vehicle)
    return camera

def add_lidar(world, blueprint, vehicle, transform):
    lidar_bp = blueprint.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('rotation_frequency', str(config['lidar']['rpm']))
    lidar = world.spawn_actor(lidar_bp, transform, attach_to=vehicle)
    return lidar
    
def add_imu(world, blueprint, vehicle, transform):
    imu_bp = blueprint.find('sensor.other.imu')
    imu_bp.set_attribute('sensor_tick', str(1./config['imu']['fps']))
    imu = world.spawn_actor(imu_bp, transform, attach_to=vehicle)
    return imu

def add_gnss(world, blueprint, vehicle, transform):
    gnss_bp = blueprint.find('sensor.other.gnss')
    gnss_bp.set_attribute('sensor_tick', str(1./config['gnss']['fps']))
    gnss = world.spawn_actor(gnss_bp, transform, attach_to=vehicle)
    return gnss
    
def set_weather(world, weather):
    world.set_weather(weather)
    return weather