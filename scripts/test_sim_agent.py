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
import numpy as np

global_img = None
sys.path.append('/home/wang/CARLA_0.9.9.4/PythonAPI/carla')
from agents.navigation.basic_agent import BasicAgent
 
def image_callback(data):
    global global_img
    array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8")) 
    array = np.reshape(array, (data.height, data.width, 4)) # RGBA format
    array = array[:, :, :3] # Take only RGB
    array = array[:, :, ::-1] # BGR
    global_img = array

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
    
    spawn_points = world_map.get_spawn_points()

    destination = spawn_points[0]
    MAX_SPEED = 30
    agent = BasicAgent(vehicle, target_speed=MAX_SPEED)
    agent.set_destination((destination.location.x,
                           destination.location.y,
                           destination.location.z))
    sensor_dict = {
        'camera':{
            'transform':carla.Transform(carla.Location(x=1.5, y=0.0, z=2.0)),
            'callback':image_callback,
            },
    }
    sm = SensorManager(world, blueprint, vehicle, sensor_dict)
    sm.init_all()
    time.sleep(0.3)
        
    while True:
        if vehicle.is_at_traffic_light():
            traffic_light = vehicle.get_traffic_light()
            if traffic_light.get_state() == carla.TrafficLightState.Red:
                traffic_light.set_state(carla.TrafficLightState.Green)
        control = agent.run_step()
        control.manual_gear_shift = False
        vehicle.apply_control(control)
        cv2.imshow('Raw image', global_img)
        cv2.waitKey(16)
        
    vehicle.destroy()
        
if __name__ == '__main__':
    main()