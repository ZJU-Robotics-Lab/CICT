#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@brief: Interface to manage all sensors in CARLA 0.9.9.4
@author: Wang Yunkai
@date: 2020.7.26
@e-mail:wangyunkai@zju.edu.cn
"""

from utils import debug, Singleton

class SensorManager(Singleton):
    def __init__(self, param_dict):
        self.param_dict = param_dict
        self.sensor_dict = {}

        self.known_sensors = ['camera', 'lidar', 'imu', 'gps']

    def init(self, key):
        if key in self.param_dict:
            sensor_type = self.get_type(key)
            if sensor_type == 'camera':
                sensor = None
                self.sensor_dict[key] = sensor
            elif sensor_type == 'lidar':
                sensor = None
                sensor.start()
                self.sensor_dict[key] = sensor
            elif sensor_type == 'imu':
                sensor = None
                sensor.start()
                self.sensor_dict[key] = sensor
            elif sensor_type == 'gps':
                sensor = None
                sensor.start()
                self.sensor_dict[key] = sensor
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