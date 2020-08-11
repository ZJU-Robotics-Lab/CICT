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

import cv2
import random
import numpy as np
import PIL.Image as Image
from PIL import ImageDraw

MAX_SPEED = 20
scale = 12
x_offset = 800
y_offset = 1000
    
def get_random_destination(spawn_points):
    return random.sample(spawn_points, 1)[0]
    
def get_map(waypoint_tuple_list):
    print(len(waypoint_tuple_list))
    origin_map = np.zeros((6000, 6000, 3), dtype="uint8")
    origin_map.fill(255)
    origin_map = Image.fromarray(origin_map)
    
    for i in range(len(waypoint_tuple_list)):
        _x1 = waypoint_tuple_list[i][0].transform.location.x
        _y1 = waypoint_tuple_list[i][0].transform.location.y
        _x2 = waypoint_tuple_list[i][1].transform.location.x
        _y2 = waypoint_tuple_list[i][1].transform.location.y

        x1 = scale*_x1+x_offset
        x2 = scale*_x2+x_offset
        y1 = scale*_y1+y_offset
        y2 = scale*_y2+y_offset
        draw = ImageDraw.Draw(origin_map)
        draw.line((x1, y1, x2, y2), 'red', width=12)
    
    return origin_map

def draw_route(agent, destination, origin_map):
    start_waypoint = agent._map.get_waypoint(agent._vehicle.get_location())
    end_waypoint = agent._map.get_waypoint(destination.location)

    route_trace = agent._trace_route(start_waypoint, end_waypoint)
    route_trace_list = []
    for i in range(len(route_trace)):
        x = scale*route_trace[i][0].transform.location.x+x_offset
        y = scale*route_trace[i][0].transform.location.y+y_offset
        route_trace_list.append(x)
        route_trace_list.append(y)
    draw = ImageDraw.Draw(origin_map)
    draw.line(route_trace_list, 'red', width=20)
    return origin_map

def get_nav(vehicle, plan_map):
    x = int(scale*vehicle.get_location().x + x_offset)
    y = int(scale*vehicle.get_location().y + y_offset)
    _nav = plan_map.crop((x-400,y-400, x+400, y+400))
    
    #r = 10
    #draw = ImageDraw.Draw(_nav)
    #draw.ellipse((_nav.size[0]//2-r, _nav.size[1]//2-r, _nav.size[0]//2+r, _nav.size[1]//2+r), fill='green', outline='green', width=10)
    
    im_rotate = _nav.rotate(vehicle.get_transform().rotation.yaw+90)
    nav = im_rotate.crop((_nav.size[0]//2-100, _nav.size[1]//2-2*80, _nav.size[0]//2+100, _nav.size[1]//2))
    nav = cv2.cvtColor(np.asarray(nav), cv2.COLOR_BGR2RGB)
    return nav

def replan(agent, destination, origin_map):
    agent.set_destination((destination.location.x,
                           destination.location.y,
                           destination.location.z))
    plan_map = draw_route(agent, destination, origin_map)
    return plan_map
    
def close2dest(vehicle, destination):
    return destination.location.distance(vehicle.get_location()) < 20
    
def main():
    client = carla.Client(config['host'], config['port'])
    client.set_timeout(config['timeout'])
    
    world = client.load_world('Town01')
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
    waypoint_tuple_list = world_map.get_topology()
    origin_map = get_map(waypoint_tuple_list)
    origin_map.resize((2000,2000)).show()
    agent = BasicAgent(vehicle, target_speed=MAX_SPEED)
    
    destination = get_random_destination(spawn_points)
    plan_map = replan(agent, destination, origin_map)
    
    """
    while True:
        if close2dest(vehicle, destination):
            destination = get_random_destination(spawn_points)
            plan_map = replan(agent, destination, origin_map)
            
        if vehicle.is_at_traffic_light():
            traffic_light = vehicle.get_traffic_light()
            if traffic_light.get_state() == carla.TrafficLightState.Red:
                traffic_light.set_state(carla.TrafficLightState.Green)
                
        control = agent.run_step()
        control.manual_gear_shift = False
        vehicle.apply_control(control)
        
        nav = get_nav(vehicle, plan_map)
        cv2.imshow('Nav', nav)
        cv2.waitKey(16)
    """
    cv2.destroyAllWindows()
    vehicle.destroy()
        
if __name__ == '__main__':
    main()