# -*- coding: utf-8 -*-
import socket
import cv2
import base64
#from time import time
from time import sleep
#import threading

port = 8080
resize = 0.8#0.8
#fps = 15
address = 'localhost'#'47.100.35.26'

def sendData():
    if cam.isOpened(): 
        success,image=cam.read()# read camera
        res = cv2.resize(image,None,fx=resize, fy=resize, interpolation = cv2.INTER_CUBIC)
        ret, jpeg=cv2.imencode('.jpg', res)
        data = jpeg.tobytes()
        if len(data) > 65535:
        	print(str(len(data)), ' > 65535')
        	res = cv2.resize(image,None,fx=resize/2, fy=resize/2, interpolation = cv2.INTER_CUBIC)
	        ret, jpeg=cv2.imencode('.jpg', res)
	        data = jpeg.tobytes()

        ret = clisocket.sendto(data,(address,port))
        return ret

    else:
        cam.release()
        clisocket.close()
        print("Service Exit !!!")
        return
        
    #t = threading.Timer(1 / fps, sendData)
    #t.start() 

if __name__ == '__main__':
    cam = cv2.VideoCapture(0)
    ret = cam.set(4,320)
    clisocket = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    #t = threading.Timer(1 / fps, sendData)
    #t.start()
    while True:
        #time1 = time()
        data_len = sendData()
        #time2 = time()
        #print(data_len, 'Use:', round(1000*(time2 - time1),3), 'ms')
        #sleep(0.1)