
import os
import sys
from os.path import join, dirname
sys.path.insert(0, join(dirname(__file__), '..'))

import simulator
simulator.load('/home/wang/CARLA_0.9.9.4')
from simulator import config
import carla
import numpy as np
import argparse
import time
import cv2
from tqdm import tqdm

from ff_collect_pm_data import sensor_dict
from ff.collect_ipm import InversePerspectiveMapping
from ff.carla_sensor import Sensor, CarlaSensorMaster

longitudinal_length = 25.0 # [m]

# for GaussianBlur
ksize = 21

parser = argparse.ArgumentParser(description='Params')
parser.add_argument('-d', '--data', type=int, default=8, help='data index')
parser.add_argument('-n', '--num', type=int, default=100000, help='total number')
args = parser.parse_args()
data_index = args.data

save_path = '/media/wang/DATASET/CARLA/town02/'+str(data_index)+'/'

def mkdir(path):
    if not os.path.exists(save_path+path):
        os.makedirs(save_path+path)
        
mkdir('ipm/')

def read_pm_time_stamp(dir_path):
    img_name_list = os.listdir(dir_path)
    time_stamp_list = []
    for img_name in img_name_list:
        time_stamp_list.append( eval(img_name.split('.png')[0]) )
    time_stamp_list.sort()
    return time_stamp_list

def read_image(time_stamp):
    img_path = save_path + 'pm/'
    file_name = str(time_stamp) + '.png'
    image = cv2.imread(img_path + file_name)
    return image


def read_state():
    state_path = save_path + 'state/'

    # read pose
    pose_file = state_path + 'pos.txt'
    time_stamp_list = []
    time_stamp_pose_dict = dict()
    file = open(pose_file, 'r') 
    while 1:
        line = file.readline()
        if not line:
            break
        if line == '\n':
            continue
        # print(line)

        line_list = line.split()

        index = eval(line_list[0])

        transform = carla.Transform()
        transform.location.x = eval(line_list[1])
        transform.location.y = eval(line_list[2])
        transform.location.z = eval(line_list[3])
        transform.rotation.pitch = eval(line_list[4])
        transform.rotation.yaw = eval(line_list[5])
        transform.rotation.roll = eval(line_list[6])

        time_stamp_list.append(index)
        time_stamp_pose_dict[index] = transform

    file.close()

    return time_stamp_list, time_stamp_pose_dict

class Param(object):
    def __init__(self):
        self.longitudinal_length = longitudinal_length
        self.ksize = ksize
        self.longitudinal_length = longitudinal_length
       
def read_pcd(time_stamp):
    pcd_path = save_path + 'pcd/'
    file_name = str(time_stamp) + '.npy'
    pcd = np.load(pcd_path+file_name)
    return pcd
    
def get_cost_map(img, point_cloud):
    width=400
    height=200
    
    img2 = np.zeros((height, width), np.uint8)
    img2.fill(255)
    #res = np.where((point_cloud[2] > -1.95)) 
    #point_cloud = point_cloud[:, res[0]]
    
    pixs_per_meter = height/longitudinal_length
    u = (height-point_cloud[0]*pixs_per_meter).astype(int)
    v = (-point_cloud[1]*pixs_per_meter+width//2).astype(int)
    
    mask = np.where((u >= 0)&(u < height))[0]
    u = u[mask]
    v = v[mask]
    
    mask = np.where((v >= 0)&(v < width))[0]
    u = u[mask]
    v = v[mask]

    img2[u,v] = 0
    
    kernel = np.ones((17,17),np.uint8)  
    img2 = cv2.erode(img2,kernel,iterations = 1)
    
    img = cv2.addWeighted(img,0.7,img2,0.3,0)
    kernel_size = (17, 17)
    sigma = 21
    img = cv2.GaussianBlur(img, kernel_size, sigma);
    return img

def main():
    time_stamp_list = read_pm_time_stamp(save_path+'pm/')

    param = Param()
    sensor = Sensor(sensor_dict['camera']['transform'], config['camera'])
    sensor_master = CarlaSensorMaster(sensor, sensor_dict['camera']['transform'], binded=True)
    inverse_perspective_mapping = InversePerspectiveMapping(param, sensor_master)

    start = 0
    end = len(time_stamp_list)
    
    for i in tqdm(range(start, end, 3)):
        try:
            time_stamp = time_stamp_list[i]
            pm_image = read_image(time_stamp)
            pcd = read_pcd(time_stamp)
    
            #tick1 = time.time()
            ipm_image = inverse_perspective_mapping.getIPM(pm_image)
            #tick2 = time.time()
    
            img = get_cost_map(ipm_image, pcd)
            cv2.imwrite(save_path+'ipm/'+str(time_stamp)+'.png', img)
            #cv2.imshow('ipm_image', img)
            #cv2.waitKey(16)
            #print('time total: ' + str(tick2-tick1))
        except:
            pass


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        #exit(0)
        pass