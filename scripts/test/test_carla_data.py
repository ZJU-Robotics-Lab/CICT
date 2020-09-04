#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import glob
import random
import numpy as np
import matplotlib.pyplot as plt

def angle_normal(angle):
    while angle >= np.pi:
        angle -= 2*np.pi
    while angle <= -np.pi:
        angle += 2*np.pi
    return angle

class CostMapDataset():
    def __init__(self, data_index = [1], dataset_path='/media/wang/DATASET/CARLA_HUMAN/town01/'):
        self.data_index = data_index
        self.max_dist = 25.
        self.max_t = 3.

        self.dataset_path = dataset_path
        self.pose_dict = {}
        self.vel_dict = {}
        self.acc_dict = {}
        self.files_dict = {}
        self.total_len = 0
        
        for index in self.data_index:
            self.read_pose(index)
            self.read_vel(index)
            self.read_acc(index)
            self.read_img(index)
        
    def read_pose(self, index):
        file_path = self.dataset_path+str(index)+'/state/pos.txt'
        ts_dict = {}
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                sp_line = line.split()
                ts = sp_line[0]
                x = float(sp_line[1])
                y = float(sp_line[2])
                z = float(sp_line[3])
                yaw = float(sp_line[5])
                ts_dict[ts] = [x, y, z, yaw]
        self.pose_dict[index] = ts_dict
        
    def read_vel(self, index):
        file_path = self.dataset_path+str(index)+'/state/vel.txt'
        ts_dict = {}
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                sp_line = line.split()
                ts = sp_line[0]
                vx = float(sp_line[1])
                vy = float(sp_line[2])
                vz = float(sp_line[3])
                ts_dict[ts] = [vx, vy, vz]
        self.vel_dict[index] = ts_dict
        
    def read_acc(self, index):
        file_path = self.dataset_path+str(index)+'/state/acc.txt'
        ts_dict = {}
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                sp_line = line.split()
                ts = sp_line[0]
                ax = float(sp_line[1])
                ay = float(sp_line[2])
                az = float(sp_line[3])
                ts_dict[ts] = [ax, ay, az]
        self.acc_dict[index] = ts_dict
    
    def read_img(self, index):
        #files = glob.glob(self.dataset_path+str(index)+'/ipm/*.png')
        files = glob.glob(self.dataset_path+str(index)+'/img/*.png')
        file_names = []
        for file in files:
            file_name = file.split('/')[-1][:-4]
            file_names.append(file_name)
        file_names.sort()
        self.files_dict[index] = file_names

    def tf_pose(self, data_index, ts, yaw, x_0, y_0):
        x_t = self.pose_dict[data_index][ts][0]
        y_t = self.pose_dict[data_index][ts][1]
        dx = x_t - x_0
        dy = y_t - y_0
        x = np.cos(yaw)*dx + np.sin(yaw)*dy
        y = np.cos(yaw)*dy - np.sin(yaw)*dx
        return x, y

    def get(self):
        data_index = random.sample(self.data_index, 1)[0]
        file_name = random.sample(self.files_dict[data_index][200:-150], 1)[0]
        
        x_0 = self.pose_dict[data_index][file_name][0]
        y_0 = self.pose_dict[data_index][file_name][1]
        yaw = np.deg2rad(self.pose_dict[data_index][file_name][3])
        
        self.files_dict[data_index].sort()
        ts_index = self.files_dict[data_index].index(file_name)
        
        ts_list = []
        x_list = []
        y_list = []
        vx_list = []
        vy_list = []
        ax_list = []
        ay_list = []
        a_list = []
        angle_list = []
        for i in range(ts_index, len(self.files_dict[data_index])-100):
            ts = self.files_dict[data_index][i]
            if float(ts)-float(file_name) > self.max_t:
                break
            else:
                ts_list.append(float(ts)-float(file_name))
                
                x_, y_ = self.tf_pose(data_index, ts, yaw, x_0, y_0)
                x_list.append(x_)
                y_list.append(y_)
            
                # vx, vy
                _vx = self.vel_dict[data_index][ts][0]
                _vy = self.vel_dict[data_index][ts][1]
                vx = np.cos(yaw)*_vx + np.sin(yaw)*_vy
                vy = np.cos(yaw)*_vy - np.sin(yaw)*_vx
                vx_list.append(vx)
                vy_list.append(vy)
                
                # ax, ay
                _ax = self.acc_dict[data_index][ts][0]
                _ay = self.acc_dict[data_index][ts][1]
                ax = _ax*np.cos(yaw) + _ay*np.sin(yaw)
                ay = _ay*np.cos(yaw) - _ax*np.sin(yaw)
                ax_list.append(ax)
                ay_list.append(ay)
                
                theta_a = np.arctan2(_ay, _ax)
                theta_v = np.arctan2(_vy, _vx)
                angle_list.append(angle_normal(theta_v-yaw))
                sign = np.sign(np.cos(theta_a-theta_v))
                a = sign*np.sqrt(ax*ax + ay*ay)
                a_list.append(a)
        return ts_list, x_list, y_list, vx_list, vy_list, ax_list, ay_list, a_list, angle_list
                

if __name__ == '__main__':
    dataset = CostMapDataset(data_index = [1,2,3,4,5,6,7,8,9])
    avg_step_list = []
    for _ in range(1000):
        ts_list, x_list, y_list, vx_list, vy_list, ax_list, ay_list, a_list, angle_list = dataset.get()
        for i in range(len(ts_list)-1):
            t2 = float(ts_list[i+1])
            t1 = float(ts_list[i])
            avg_step_list.append(1000*(t2-t1))
    print(round(sum(avg_step_list)/len(avg_step_list),2), 'ms')
        
    """
    max_x = 25.
    max_y = 25.
    max_speed = 12.0

    fig = plt.figure(figsize=(15,15))
    ax1 = fig.add_subplot(111)

    ax1.plot(x_list, y_list, label='trajectory', color = 'b', linewidth=8)
    ax1.set_xlabel('Tangential/(m)')
    ax1.set_ylabel('Normal/(m)')
    ax1.set_xlim([0., max_x])
    ax1.set_ylim([-max_y, max_y])
    plt.legend(loc='lower right')

    vx = np.array(vx_list)
    vy = np.array(vy_list)
    v = np.sqrt(np.power(vx, 2), np.power(vy, 2))
    angle = np.arctan2(vy, vx)/np.pi*max_speed
    ax2 = ax1.twinx()
    ts_list = [(20./3.)*item for item in ts_list]
    ax2.plot(ts_list, v, label='speed', color = 'r', linewidth=5)
    ax2.plot(ts_list, a_list, label='acc', color = 'y', linewidth=5)
    ax2.plot(ts_list, angle_list, label='angle', color = 'g', linewidth=5)
    ax2.set_ylabel('Velocity/(m/s)')
    ax2.set_ylim([-max_speed, max_speed])
    plt.legend(loc='upper right')

    plt.show()
    """