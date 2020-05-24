# -*- coding: utf-8 -*-
from __future__ import print_function
import math
import time
import serial

class Filter():
    def __init__(self):
        self.v = 0.0
        self.yaw = 0.0

        self.x = 0.0
        self.y = 0.0
        self.last_t = time.time()
        self.cnt = 0
        self.INIT_STEPS = 50
        self.ALPHA = 0.2
        self.MAX_V = 15.0
        
        self.x_his = []
        self.y_his = []
        
        self.x_bias = 0.0
        self.y_bias = 0.0
        
    
    def dist(self, x, y):
        return math.sqrt((self.x-x)**2 + (self.y-y)**2)
        
    def step(self, x, y):
        if self.cnt >= self.INIT_STEPS:
            x = x - self.x_bias
            y = y - self.y_bias
            
        self.cnt += 1
        now = time.time()

        d = self.dist(x, y)
        dt = now - self.last_t
        if dt < 0.0001: dt = 0.1
        v = d/dt
        
        if self.cnt < self.INIT_STEPS-10:
            self.x = (1-self.ALPHA)*self.x + self.ALPHA*x
            self.y = (1-self.ALPHA)*self.y + self.ALPHA*y
            self.x_bias = self.x
            self.y_bias = self.y
            
        elif self.cnt < self.INIT_STEPS:
            self.x = (1-self.ALPHA)*self.x + self.ALPHA*x
            self.y = (1-self.ALPHA)*self.y + self.ALPHA*y
            self.x_his.append(x)
            self.y_his.append(y)
        else:
            if v > self.MAX_V:
                self.x = self.x + math.cos(self.yaw)*self.v*dt
                self.y = self.y + math.sin(self.yaw)*self.v*dt
            else:
                self.v = (1-self.ALPHA)*self.v + self.ALPHA*v
                self.x = x
                self.y = y
                
                self.yaw = math.atan2((y-self.y_his[5]), (x-self.y_his[5]))
        
            self.x_his.pop(0)
            self.y_his.pop(0)
            self.x_his.append(self.x)
            self.y_his.append(self.y)
            self.last_t = now
        
        return self.x, self.y, self.v
    
class GPS():
    def __init__(self, port = '/dev/ttyUSB0'):
        self.serial = None
        self.port = port
        self.filter = Filter()
    
    def start(self):
        try:
            self.serial = serial.Serial(self.port, 115200)
        except:
            print('Error when open GPS')
    
    def close(self):
        self.serial.close()
        
    def get(self):
        data = self.serial.readline()
        while data[:6] != '$GPGGA':
            data = self.serial.readline()
        x, y, v = self.parseGPS(data)
        return x, y, v
    
    def parseGPS(self, line):
        try:
            data = line.split(',')
            #print(data)
            latitude = data[2]	
            longtitude = data[4]
            #num_star = data[7]
            #hdop = data[8]
            #convert degree+minute to degree
        	#lantitude: DDmm.mm
        	#longtitude: DDDmm.mm
            lan_degree = latitude[:2]
            lan_minute = latitude[2:]
            latitude = float(lan_degree) + float(lan_minute)/60
        
            long_degree = longtitude[:3]
            long_minute = longtitude[3:]
            longtitude = float(long_degree) + float(long_minute)/60
            
            x, y = self.gps2xy(self, latitude, longtitude)
            filter_x, filter_y, filter_v = self.filter.step(x, y)
            return filter_x, filter_y, filter_v
        except:
            #print('Error when parse GPS:', data)
            return 0.0, 0.0
        
    
    def gps2xy(self, latitude, longtitude):
    
    	#remain to be done!!! figure out the formula meanings!!!
    	latitude = latitude * math.pi/180
    	longtitude = longtitude *math.pi/180
    
    	#the radius of the equator
    	radius = 6378137
    	#distance of the two poles
    	distance = 6356752.3142
    	#reference??
    	base = 30 * math.pi/180
    	
    	radius_square = pow(radius,2)
    	distance_square = pow(distance,2)
    	
    	e = math.sqrt(1 - distance_square/radius_square)
    	e2 = math.sqrt(radius_square/distance_square - 1)
    
    	cosb0 = math.cos(base)
    	N = (radius_square / distance) / math.sqrt( 1+ pow(e2,2)*pow(cosb0,2))
    	K = N*cosb0
    	
    	sinb = math.sin(latitude)
    	tanv = math.tan(math.pi/4 + latitude/2)
    	E2 = pow((1 - e*sinb) / (1+ e* sinb),e/2)
    	xx = tanv * E2;
    	
    	xc = K * math.log(xx)
    	yc = K * longtitude
    	return xc,yc