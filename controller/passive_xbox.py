#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pygame
import platform
import threading
sys_type = platform.system()

class JoyStick:
    def __init__(self, verbose=False, cloc_time=20):
        self.verbose = verbose
        self.cloc_time = cloc_time
        
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
        self.speed = 0.0
        self.rotation = 0.0
    
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
            self.parse_buttom()
            self.parse_hat()
            self.clock.tick(self.cloc_time)
    
    def parse_axes(self):
        #axes = self.joystick.get_numaxes()
        if sys_type == 'Windows':
            speed = self.joystick.get_axis(3)
            speed = max(-speed, 0)
            if self.verbose:
                print('speed:',speed)
            
            rotation = self.joystick.get_axis(2)
            rotation = - rotation
            if self.verbose:
                print('rotation:', rotation)
        else:
            speed = self.joystick.get_axis(4)
            speed = max(-speed, 0)
            self.speed = speed
            if self.verbose:
                print('speed:',speed)
            
            rotation_l = self.joystick.get_axis(2)
            rotation_r = self.joystick.get_axis(5)
            # fix some bug in xbox reading
            if abs(rotation_l-0)<0.0001:
                rotation_l = -1.0
            if abs(rotation_r-0)<0.0001:
                rotation_r = -1.0  

            rotation = (rotation_l - rotation_r)/2
            rotation = - rotation
            self.rotation = rotation
            if self.verbose:
                print('rotation:', rotation)
        
    def get(self):
        return self.speed, self.rotation
        
    def parse_buttom(self):
        buttons = self.joystick.get_numbuttons()
        for i in range(buttons):
            button = self.joystick.get_button(i)
            if i==0 and button ==1: # A
                print('Set backward')
            if i==1 and button ==1: # B
                pass
            if i==2 and button ==1: # X
                pass
            if i==3 and button ==1: # Y
                print('Set forward')
            if i==4 and button ==1: # LB
                pass
            if i==5 and button ==1: # RB
                pass
            if i==6 and button ==1: # BACK
                pass
            if i==7 and button ==1: # START
                pass
            if i==8 and button ==1: # Middle LOG
                if self.halt == True:
                    print('Stop !!!')
            if i==9 and button ==1: # Left Axis
                pass
            if i==10 and button ==1: # Right Axis
                pass
            
    def parse_hat(self):
        hats = self.joystick.get_numhats()
        for i in range(hats):
            hat = self.joystick.get_hat(i)
            if hat==(1,0): # FX right
                acc_time = 0
                acc_time -= 1.5
                print('Set acc_time:', acc_time)
            if hat==(-1,0): # FX left
                acc_time = 0
                acc_time += 1.5
                print('Set acc_time:', acc_time)
            if hat==(0,1): # FX up
                max_speed = 0
                max_speed += 100
                print('Set max_speed:', max_speed)
            if hat==(0,-1): # FX down
                max_speed = 0
                max_speed -= 100
                print('Set max_speed:', max_speed)


if __name__ == '__main__':
    joystick = JoyStick(verbose=False)
    joystick.start()
