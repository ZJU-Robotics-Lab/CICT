#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import glob

fps = 150
video_size = (640,360)

videoWriter = cv2.VideoWriter("total3.mp4", cv2.VideoWriter_fourcc(*'MJPG'), fps, video_size)

files = glob.glob('test_output/*.png')
img_ts_list = []
for file in files:
    ts = file.split('/')[1][:-4]
    img_ts_list.append(ts)
    
img_ts_list.sort()
print('total:', len(img_ts_list))
for img_name in img_ts_list:
    print(img_ts_list.index(img_name)/len(img_ts_list))

    img = cv2.imread('test_output/'+img_name+'.png')
    img = cv2.resize(img, (640,360))
    videoWriter.write(img)
  
    
videoWriter.release()