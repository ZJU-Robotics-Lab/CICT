#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2

fps = 150
video_size = (600, 300)

videoWriter = cv2.VideoWriter("total.mp4", cv2.VideoWriter_fourcc(*'MJPG'), fps, video_size)

for num in range(800, 7000):
    try:
        img = cv2.imread("images/"+str(num)+".png")
        videoWriter.write(img)
    except:
        print('Error in', num)    
    
videoWriter.release()