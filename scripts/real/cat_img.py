#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import glob
from os.path import join, dirname
sys.path.insert(0, join(dirname(__file__), '..'))
sys.path.insert(0, join(dirname(__file__), '../..'))

import cv2
import numpy as np

def read_files(index):
    file_path = '/media/wang/Data/video/data'+str(index)
    img_list = []
    out_list = []
    pcd_list = []
    nav_list = []
    cost_list = []
    for file_name in glob.glob(file_path+'/img/'+'*.png'):
        img_list.append(file_name.split('/')[-1][:-4])
    for file_name in glob.glob(file_path+'/output/'+'*.png'):
        out_list.append(file_name.split('/')[-1][:-4])
    for file_name in glob.glob(file_path+'/lidar/'+'*.npy'):
        pcd_list.append(file_name.split('/')[-1][:-4])
    for file_name in glob.glob(file_path+'/nav/'+'*.png'):
        nav_list.append(file_name.split('/')[-1][:-4])
    for file_name in glob.glob(file_path+'/cost/'+'*.png'):
        cost_list.append(file_name.split('/')[-1][:-4])
    img_list.sort(), pcd_list.sort(), nav_list.sort(), cost_list.sort(), out_list.sort()
    return img_list, pcd_list, nav_list, cost_list, out_list



def find_nn(ts, ts_list, back=0):
    dt_list = list(map(lambda x: abs(float(x)-float(ts)), ts_list))
    index = max(0, dt_list.index(min(dt_list)) - back)
    return ts_list[index]

if __name__ == '__main__':
    rate = 1.25
    rate2 = 1.0
    dataset = {}
    

    fps = 30
    video_size = (1280, 720)

    videoWriter = cv2.VideoWriter("/media/wang/Data/video/first-person/2.mp4", cv2.VideoWriter_fourcc(*'MJPG'), fps, video_size)

    for index in [2,4,5]:
        img_list, pcd_list, nav_list, cost_list, out_list = read_files(index)
        dataset[index] = {'img_list':img_list, 'pcd_list':pcd_list, 'nav_list':nav_list, 'cost_list':cost_list, 'out_list':out_list}

    for index in [2]:
        choose_dataset = dataset[index]
        for ts in choose_dataset['img_list']:
            img = cv2.imread('/media/wang/Data/video/data'+str(index)+'/output/'+ts+'.png')
            #print(img.shape) #(720, 1280, 3)
            if img is None: continue
            #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            nav_ts = find_nn(ts, choose_dataset['nav_list'])
            cost_ts = find_nn(ts, choose_dataset['cost_list'])
            nav = cv2.imread('/media/wang/Data/video/data'+str(index)+'/nav/'+nav_ts+'.png')
            costmap = cv2.imread('/media/wang/Data/video/data'+str(index)+'/cost/'+cost_ts+'.png')
            nav = cv2.cvtColor(nav, cv2.COLOR_BGR2RGB) #(160, 200, 3)
            #input_img = get_img(img, nav)
            nav = cv2.resize(nav, (int(200*rate), int(rate*160)))
            img[0:int(rate*160), -int(200*rate):] = nav
            img[0:int(rate2*200), 0:int(400*rate2)] = costmap
            cv2.imshow('img', img)
            videoWriter.write(img)
            #cv2.imshow('costmap', costmap)
            cv2.waitKey(1)
            
    cv2.destroyAllWindows()
    videoWriter.release()











        