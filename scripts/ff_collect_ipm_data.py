
import os
import sys
from os.path import join, dirname
sys.path.insert(0, join(dirname(__file__), '..'))

import simulator
simulator.load('/home/zdk/CARLA_0.9.9.4')
from simulator import config
import carla
import numpy as np
import argparse
import time


longitudinal_length = 40.0 # [m]

# for GaussianBlur
ksize = 9

parser = argparse.ArgumentParser(description='Params')
parser.add_argument('-d', '--data', type=int, default=1, help='data index')
parser.add_argument('-n', '--num', type=int, default=100000, help='total number')
args = parser.parse_args()
data_index = args.data

# save_path = '/home/zdk/DATASET/CARLA/'+str(data_index)+'/'
# sensor_dict = {
#     'camera':{
#         'transform':carla.Transform(carla.Location(x=1.5, y=0.0, z=2.0)),
#         # 'callback':image_callback,
#     },
#     'lidar':{
#         'transform':carla.Transform(carla.Location(x=1.5, y=0.0, z=2.0)),
#         # 'callback':lidar_callback,
#     },
# }

from ff_collect_pm_data import save_path, sensor_dict, mkdir

mkdir('ipm/')



from ff.system import printVariable
from ff.system import env_path
# env_path.remove_python2_path(sys)
import cv2
# env_path.append_python2_path(sys)

from ff.collect_ipm import InversePerspectiveMapping
from ff.carla_sensor import Sensor, CarlaSensorMaster
from sensor_msgs.msg import Image
from ff.carla_sensor import CarlaSensorDataConversion

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
    # print(img.shape)
    return image


class Param(object):
    def __init__(self):
        self.longitudinal_length = longitudinal_length
        self.ksize = ksize
        self.longitudinal_length = longitudinal_length




def main():
    time_stamp_list = read_pm_time_stamp(save_path+'pm/')

    param = Param()
    sensor = Sensor(sensor_dict['camera']['transform'], config['camera'])
    sensor_master = CarlaSensorMaster(sensor, sensor_dict['camera']['transform'], binded=True)
    inverse_perspective_mapping = InversePerspectiveMapping(param, sensor_master)

    # pm_image_pub = rospy.Publisher('~pm_image', Image, queue_size=1)
    # ipm_image_pub = rospy.Publisher('~ipm_image', Image, queue_size=1)

    for time_stamp in time_stamp_list:
        pm_image = read_image(time_stamp)

        tick1 = time.time()
        ipm_image = inverse_perspective_mapping.getIPM(pm_image)
        tick2 = time.time()
        # print('time total: ' + str(tick2-tick1))
        printVariable('time total', tick2-tick1)
        print()

        # pm_image_pub.publish(CarlaSensorDataConversion.cv2ImageToSensorImage(pm_image, 'pm_image'))
        # ipm_image_pub.publish(CarlaSensorDataConversion.cv2ImageToSensorImage(ipm_image, 'ipm_image'))






# import rospy


if __name__ == '__main__':
    # rospy.init_node('collect_ipm')
    try:
        main()
    except KeyboardInterrupt:
        exit(0)