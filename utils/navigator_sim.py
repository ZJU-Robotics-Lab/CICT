#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import random
import numpy as np
import PIL.Image as Image
from PIL import ImageDraw

scale = 12.0
x_offset = 800#2500
y_offset = 1000#3000

def get_random_destination(spawn_points):
    return random.sample(spawn_points, 1)[0]
    
def get_map(waypoint_tuple_list):
    origin_map = np.zeros((6000, 6000, 3), dtype="uint8")
    origin_map.fill(255)
    origin_map = Image.fromarray(origin_map)
    """
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
        draw.line((x1, y1, x2, y2), 'white', width=12)
    """
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
    draw.line(route_trace_list, 'red', width=30)
    return origin_map

def get_nav(vehicle, plan_map):
    x = int(scale*vehicle.get_location().x + x_offset)
    y = int(scale*vehicle.get_location().y + y_offset)
    _nav = plan_map.crop((x-400,y-400, x+400, y+400))
    
    #r = 10
    #draw = ImageDraw.Draw(_nav)
    #draw.ellipse((_nav.size[0]//2-r, _nav.size[1]//2-r, _nav.size[0]//2+r, _nav.size[1]//2+r), fill='green', outline='green', width=10)
    
    im_rotate = _nav.rotate(vehicle.get_transform().rotation.yaw+90)
    nav = im_rotate.crop((_nav.size[0]//2-150, _nav.size[1]//2-2*120, _nav.size[0]//2+150, _nav.size[1]//2))
    nav = cv2.cvtColor(np.asarray(nav), cv2.COLOR_BGR2RGB)
    return nav

def get_big_nav(vehicle, plan_map):
    x = int(scale*vehicle.get_location().x + x_offset)
    y = int(scale*vehicle.get_location().y + y_offset)
    _nav = plan_map.crop((x-400,y-400, x+400, y+400))
    
    r = 20
    draw = ImageDraw.Draw(_nav)
    draw.ellipse((_nav.size[0]//2-r, _nav.size[1]//2-r, _nav.size[0]//2+r, _nav.size[1]//2+r), fill='green', outline='green', width=10)
    
    im_rotate = _nav.rotate(vehicle.get_transform().rotation.yaw+90)
    #nav = im_rotate
    nav = im_rotate.crop((0, 0, _nav.size[0], _nav.size[1]//2))
    nav = cv2.cvtColor(np.asarray(nav), cv2.COLOR_BGR2RGB)
    return nav

def replan(agent, destination, origin_map):
    agent.set_destination((destination.location.x,
                           destination.location.y,
                           destination.location.z))
    plan_map = draw_route(agent, destination, origin_map)
    return plan_map
    
def close2dest(vehicle, destination):
    return destination.location.distance(vehicle.get_location()) < 30