#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@brief: Interface to manage all sensors
@author: Wang Yunkai
@date: 2019.11.28
@e-mail:wangyunkai@zju.edu.cn
"""

from utils import debug, Singleton

from robot_camera import Camera
from LiDAR import LiDAR

class SensorManager(Singleton):
    def __init__(self, param_dict):
        self.param_dict = param_dict
        self.sensor_dict = {}

        self.known_sensors = ['camera', 'lidar', 'imu', 'gps']

    def init(self, key):
        if key in self.param_dict:
            sensor_type = self.get_type(key)
            if sensor_type == 'camera':
                sensor = Camera()
                self.sensor_dict[key] = sensor
            elif sensor_type == 'lidar':
                sensor = LiDAR()
                sensor.start()
                self.sensor_dict[key] = sensor
            elif sensor_type == 'imu':
                pass
                self.sensor_dict[key] = sensor_type
            elif sensor_type == 'gps':
                pass
                self.sensor_dict[key] = sensor_type
            else:
                debug(info=str(key)+' not initialized', info_type='warning')
        else:
            debug(info='Unknown sensor '+ str(key), info_type='error')
            return None	

    def init_all(self):
        for key in self.param_dict:
            try:
                self.init(key)
            except:
                debug(info=str(key)+' initialize failed', info_type='error')

    def close_all(self):
        for key in self.param_dict:
            try:
                self.sensor_dict[key].close()
                debug(info=str(key)+' closed', info_type='success')
            except:
                debug(info=str(key)+' has no attribute called \'close\'', info_type='message')
        
    def __del__(self):
        self.close_all()
    
	# get sensor instance
    def __getitem__(self, key):
        if key in self.sensor_dict:
            return self.sensor_dict[key]
        else:
            debug(info='No sensor called '+ str(key), info_type='error')
            return None
    
    # set sensor param
    def __setitem__(self, key, value):
        if key in self.param_dict:
            self.param_dict[key] = value
            return True
        else:
            debug(info='No sensor called '+ str(key), info_type='error')
            return None

    def get_type(self, key):
        sensor_type = key.split(':')[0]
        if sensor_type in self.known_sensors:
            return sensor_type
        else:
            debug(info='Unknown sensor type '+ str(key), info_type='error')
            return None


if __name__ == '__main__':
    sensor_dict = {'camera:0':[666],
                   'lidar:0':[1,2,'sss'],
                   'lidar:1':[1,3,'aaa'],
                   'imu:0':['asd', '555',9],
                   }
    sm = SensorManager(sensor_dict)
    print(sm.param_dict)
    print(sm.sensor_dict)
    sm.init_all()
    print(sm.param_dict)
    print(sm.sensor_dict)