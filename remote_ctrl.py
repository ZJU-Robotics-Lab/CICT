#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@brief: remote control
@author: Wang Yunkai
@date: 2020.5.11
@e-mail:wangyunkai@zju.edu.cn
"""
import cv2
import json
import random
from time import sleep

from informer import Informer, config
from sensor_manager import SensorManager
from controller import Controller

config.PUBLICT_IP = '127.0.0.1'#'47.100.46.11'

class RoboCar(Informer):
    def __init__(self, robot_id=None, block=True):
        super(RoboCar, self).__init__(robot_id, block)
        
        self.ctrl = Controller()
        self.ctrl.start()
        self.ctrl.set_forward()
        self.ctrl.set_speed(0)
        self.ctrl.set_acc_time(1.0)
        
        self.hand_brake = False
        self.speed = 0.
        self.steer = 0.
        self.gear = 1
        
        
    def parse_cmd(self, cmd):
        command = json.loads(cmd['Data'])
        steer = command['w']
        speed = command['v']
        hand_brake = bool(command['h'])
        #brake = command['b']
        gear = int(command['g'])
        rate = gear/5.0

        self.ctrl.set_speed(rate * abs(speed))
        self.ctrl.set_rotation(steer)
        
        if speed * self.speed < 0:
            if speed > 0:
                self.ctrl.set_forward()
            else:
                self.ctrl.set_backward()
        
        if hand_brake:
            self.ctrl.set_stop()
            
        self.steer = steer
        self.speed = speed
        self.gear = gear
        self.hand_brake = hand_brake


if __name__ == '__main__':
    robot = RoboCar(random.randint(100000,999999), block=False)

    sensor_dict = {'camera':None,
                   #'lidar':None,
                   }
    sm = SensorManager(sensor_dict)
    sm.init_all()

    while True:
        img = sm['camera'].getImage()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        #img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        robot.send_vision(img, False)

        # send robot sensors
        v = random.random()*5
        w = random.random()*10 - 5
        c = False if random.random() > 0.3 else True
        robot.send_sensor_data(v, w, c)

        sleep(1/5.)
        
    sm.close_all()
