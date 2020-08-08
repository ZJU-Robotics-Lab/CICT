#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from os.path import join, dirname
sys.path.insert(0, join(dirname(__file__), '..'))

import simulator
simulator.load('/home/wang/CARLA_0.9.9.4')
import carla

import pygame
import threading

class JoyStick:
    def __init__(self, verbose=False, cloc_time=20):
        self.verbose = verbose
        self.cloc_time = cloc_time
        
        self.control = carla.VehicleControl()
        self.control.manual_gear_shift = False
        self.control.reverse = False
        self.control.gear = 1
        
        pygame.init()
        self.halt = False
        self.clock = pygame.time.Clock()
        pygame.joystick.init()
        
        self.joystick_count = pygame.joystick.get_count()
        if self.joystick_count == 0:
            print('No joystick found !')
        elif self.joystick_count > 1:
            print('Found joystick number:', self.joystick_count)
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        
        self.stop_read_event = threading.Event()
        self.parse_cyclic = threading.Thread(
            target=self.parse, args=()
        )
    
    def start(self):
        self.stop_read_event.clear()
        self.parse_cyclic.start()
        
    def stop(self):
        self.stop_read_event.set()   
    
    def parse(self):
        while not self.stop_read_event.is_set():
            if pygame.QUIT in pygame.event.get():
                self.stop_read_event.set()
            
            self.parse_axes()
            self.clock.tick(self.cloc_time)
    
    def parse_axes(self):
        speed = self.joystick.get_axis(4)
        speed = max(-speed, 0)
        if self.verbose:
            print('speed:',speed)
        if not self.halt:
            self.control.throttle = speed/2.0
        
        rotation_l = self.joystick.get_axis(2)
        rotation_r = self.joystick.get_axis(5)
        # fix some bug in xbox reading
        if abs(rotation_l-0)<0.0001:
            rotation_l = -1.0
        if abs(rotation_r-0)<0.0001:
            rotation_r = -1.0  

        rotation = (rotation_l - rotation_r)/2.
        rotation = - rotation
        if self.verbose:
            print('rotation:', rotation)
        if not self.halt:
            self.control.steer = rotation
                
    def get(self):
        return self.control
