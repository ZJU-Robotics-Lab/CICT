# -*- coding: utf-8 -*-
from __future__ import print_function
import math
import time
import serial
import numpy as np
import rospy
from std_msgs.msg import String
from rospy.numpy_msg import numpy_msg
#from rospy_tutorials.msg import Floats
    
class GPS():
    def __init__(self, port = '/dev/ttyUSB0'):
        self.serial = None
        self.port = port
    
    def start(self):
        try:
            self.serial = serial.Serial(self.port, 115200)
            if self.serial is not None:
                print("Open serial port successfully !")

            #self.serial.write(b'reset\r\n')
            #self.serial.write(b'com trans com2 gnsscom2\r\n')
            #self.serial.write(b'gpgga 0.1\r\n')

        except:
            print('Error when open GPS')
    
    def close(self):
        self.serial.close()
        
    def get(self):
        data = self.serial.readline()
        while data[:6] != '$GPGGA':
            data = self.serial.readline()
        parsed_data = self.parseGPS(data)
        return parsed_data
    
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
            
            x, y = self.gps2xy(latitude, longtitude)
            return str(x)+'\t'+str(y)+'\n'
        except:
            print('Error when parse GPS:', data)
            return None
        
    
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
    
if __name__ == '__main__':
    gps = GPS('/dev/ttyUSB1')
    gps.start()
    pub = rospy.Publisher('gps', String, queue_size=1)
    rospy.init_node('talker', anonymous=True)
    while True:
        data = gps.get()
        if data == None:
            continue
        print(data)
        #array = np.array(data, dtype=np.float64)
        pub.publish(data)
        
