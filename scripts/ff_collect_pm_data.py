
import os
import sys
from os.path import join, dirname
sys.path.insert(0, join(dirname(__file__), '..'))

import simulator
simulator.load('/home/wang/CARLA_0.9.9.4')
from simulator import config
import carla
import argparse
import time

MAX_SPEED = 20
TRAJ_LENGTH = 25
vehicle_width = 2.2
longitudinal_sample_number_near = 8
longitudinal_sample_number_far = 0.5
lateral_sample_number = 20
lateral_step_factor = 1.0

parser = argparse.ArgumentParser(description='Params')
parser.add_argument('-d', '--data', type=int, default=1, help='data index')
parser.add_argument('-n', '--num', type=int, default=100000, help='total number')
args = parser.parse_args()
data_index = args.data

save_path = '/media/wang/DATASET/CARLA/town02/'+str(data_index)+'/'

sensor_dict = {
    'camera':{
        'transform':carla.Transform(carla.Location(x=1.5, y=0.0, z=2.0)),
        # 'callback':image_callback,
    },
    'lidar':{
        'transform':carla.Transform(carla.Location(x=1.5, y=0.0, z=2.0)),
        # 'callback':lidar_callback,
    },
}

from ff.system import env_path
env_path.remove_python2_path(sys)
import cv2
env_path.append_python2_path(sys)

from ff.collect_pm import CollectPerspectiveImage
from ff.carla_sensor import Sensor, CarlaSensorMaster

def mkdir(path):
    if not os.path.exists(save_path+path):
        os.makedirs(save_path+path)
        
def read_img(time_stamp):
    img_path = save_path + 'img/'
    file_name = str(time_stamp) + '.png'
    img = cv2.imread(img_path + file_name)
    return img

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
    

def distance(pose1, pose2):
    return pose1.location.distance(pose2.location)

def find_traj_with_fix_length(start_index, time_stamp_list, time_stamp_pose_dict):
    length = 0.0
    for i in range(start_index, len(time_stamp_list)-1):
        length += distance(time_stamp_pose_dict[time_stamp_list[i]], time_stamp_pose_dict[time_stamp_list[i+1]])
        # print('here: '+str((i, length)))
        if length >= TRAJ_LENGTH:
            return i
    return -1


class Param(object):
    def __init__(self):
        self.traj_length = float(TRAJ_LENGTH)
        self.target_speed = float(MAX_SPEED)
        self.vehicle_width = float(vehicle_width)
        self.longitudinal_sample_number_near = longitudinal_sample_number_near
        self.longitudinal_sample_number_far = longitudinal_sample_number_far
        self.lateral_sample_number = lateral_sample_number
        self.lateral_step_factor = lateral_step_factor

def main():
    mkdir('pm/')
    time_stamp_list, time_stamp_pose_dict = read_state()
    time_stamp_list.sort()

    relative_time_stamp_list = [t - time_stamp_list[0] for t in time_stamp_list]
    print(len(relative_time_stamp_list))

    param = Param()
    sensor = Sensor(sensor_dict['camera']['transform'], config['camera'])
    sensor_master = CarlaSensorMaster(sensor, sensor_dict['camera']['transform'], binded=True)
    collect_perspective = CollectPerspectiveImage(param, sensor_master)

    for index, time_stamp in enumerate(time_stamp_list):
        end_index = find_traj_with_fix_length(index, time_stamp_list, time_stamp_pose_dict)
        if end_index < 0:
            print('current index: ' + str(index))
            continue

        vehicle_transform = time_stamp_pose_dict[time_stamp]  # in world coordinate
        traj_pose_list = []
        for i in range(index, end_index):
            time_stamp_i = time_stamp_list[i]
            time_stamp_pose = time_stamp_pose_dict[time_stamp_i]
            traj_pose_list.append((time_stamp_i, time_stamp_pose))

        img = read_img(time_stamp)
        #t1 = time.time()
        empty_image = collect_perspective.getPM(traj_pose_list, vehicle_transform, img)
        #t2 = time.time()

        #cv2.imshow('empty_image', empty_image)
        #cv2.waitKey(3)
        cv2.imwrite(save_path+'pm/'+str(time_stamp)+'.png', empty_image)
        #print('time total: ' + str(t2-t1))
        #print()

if __name__ == '__main__':
    # rospy.init_node('collect_pm')
    try:
        main()
    except KeyboardInterrupt:
        #exit(0)
        pass