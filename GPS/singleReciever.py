import os
from datetime import datetime
from math import *
import serial
import numpy as np
import matplotlib.pyplot as plt

def gps2xy(latitude, longtitude):

	#remain to be done!!! figure out the formula meanings!!!
	latitude = latitude * pi/180
	longtitude = longtitude *pi/180

	#the radius of the equator
	radius = 6378137
	#distance of the two poles
	distance = 6356752.3142
	#reference??
	base = 30 * pi/180
	
	radius_square = pow(radius,2)
	distance_square = pow(distance,2)
	
	e = sqrt(1 - distance_square/radius_square)
	e2 = sqrt(radius_square/distance_square - 1)

	cosb0 = cos(base)
	N = (radius_square / distance) / sqrt( 1+ pow(e2,2)*pow(cosb0,2))
	K = N*cosb0
	
	sinb = sin(latitude)
	tanv = tan(pi/4 + latitude/2)
	E2 = pow((1 - e*sinb) / (1+ e* sinb),e/2)
	xx = tanv * E2;
	
	xc = K * log(xx)
	yc = K * longtitude
	return xc,yc

def readgpgga(dataline):
	data = dataline.split(',')
	latitude = data[2]	
	longtitude = data[4]
	num_star = data[7]
	hdop = data[8]
	#whether the gps is trusty
	#if hdop> 1.0:
	#	return -1
	
	#convert degree+minute to degree
	#lantitude: DDmm.mm
	#longtitude: DDDmm.mm
	lan_degree = latitude[:2]
	lan_minute = latitude[2:]
	latitude = float(lan_degree) + float(lan_minute)/60

	long_degree = longtitude[:3]
	long_minute = longtitude[3:]
	longtitude = float(long_degree) + float(long_minute)/60
	return latitude,longtitude

def readgpsline(data):
	latitude,longtitude = readgpgga(data)
	xc,yc = gps2xy(latitude, longtitude)
	return xc,yc


def initialserial(port):
	if not os.path.isdir(port):
		return False
	ser = serial.Serial(port, 115200, timeout = 0.5)
	return ser	

def readfile(path):
	datafile = open(path,'r')
	data = datafile.readlines()
	coordinate = []
	for i in range(len(data)):
		xc,yc = readgpsline(data[i])
		coordinate.append([xc, yc])
	coordinate = np.asarray(coordinate)
	return coordinate

#main process
READ_ONLINE = False 
SAVE_TO_FILE = True 
if SAVE_TO_FILE:
	dt = datetime.now()
	filename = dt.strftime('%y-%m%d-%I%M%S%p')+ ".txt"
	savefile = open(filename,'w')


if READ_ONLINE:
	ser = initialserial("/dev/ttyUSB0")
	if not ser:
		print "Error! No serial signal! Quit Process !"
		quit()
	data = ser.readline()
	if SAVE_TO_FILE:
		savefile.write(data)
		savefile.write('\n')
	xc,yc = readgpsline(data)
	print xc,yc
else:
	coordinate = readfile("/home/huifang/Desktop/GPS/ReceivedTofile-COM3-12_3_2019_16-38-56.DAT")
	print "Read %d points." % coordinate.shape[0]
	plt.plot(coordinate[:,0], coordinate[:,1], linewidth = 2)	
	plt.show()
if SAVE_TO_FILE:
	savefile.close()








