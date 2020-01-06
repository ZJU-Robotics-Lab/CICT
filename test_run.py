#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@brief: test run all codes
@author: Wang Yunkai
@date: 2020.1.14
@e-mail:wangyunkai@zju.edu.cn
"""

from sensor_manager import SensorManager
from controller import Controller

if __name__ == '__main__':
    
    import cv2
    sensor_dict = {'camera:0':None,
                   }
    
    ctrl = Controller()
    
    sm = SensorManager(sensor_dict)
    sm.init_all()
    
    ctrl.start()
    ctrl.set_forward()
    ctrl.set_speed(0)
    ctrl.set_acc_time(1.0)
    ctrl.set_rotation(-0.9)
    
    for _ in range(100):
        img = sm['camera:0'].getImage()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR);
        cv2.imshow('input_image', img)
        cv2.waitKey(30)
        ctrl.set_speed(0)
        
    cv2.destroyAllWindows()
    sm.close_all()
