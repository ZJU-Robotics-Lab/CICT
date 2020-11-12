
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import glob
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import copy
from scipy.special import comb
np.set_printoptions(suppress=True, precision=4, linewidth=65535)

def expand_control_points(point_array):
    point_array_expand = copy.deepcopy(point_array)
    size = point_array.shape[1]
    assert size >= 3
    for i in range(1,size-3):
        p0, p1, p2 = point_array[:,i], point_array[:,i+1], point_array[:,i+2]
        norm1, norm2 = np.linalg.norm(p0-p1), np.linalg.norm(p2-p1)
        pc = p1 - 0.5*np.sqrt(norm1*norm2)*((p0-p1)/norm1 + (p2-p1)/norm2)
        point_array_expand[:,i+1] = pc
    return point_array_expand

def bernstein(t, i, n):
    return comb(n,i) * t**i * (1-t)**(n-i)

def bezier_curve(t, point_array, bias=0):
    t = np.clip(t, 0, 1)
    n = point_array.shape[1]-1
    p = np.array([0.,0.]).reshape(2,1)
    size = len(t) if isinstance(t, np.ndarray) else 1
    p = np.zeros((2, size))
    new_point_array = np.diff(point_array, n=bias, axis=1)
    for i in range(n+1-bias):
        p += new_point_array[:,i][:,np.newaxis] * bernstein(t, i, n-bias) * n**bias
    return p


class Bezier(object):
    def __init__(self, time_list, x_list, y_list, v0, vf=(0.000001,0.000001)):
        t0, x0, y0 = time_list[0], x_list[0], y_list[0]
        t_span = time_list[-1] - time_list[0]
        time_array = np.array(time_list)
        x_array, y_array = np.array(x_list), np.array(y_list)
        time_array -= t0
        x_array -= x0
        y_array -= y0
        time_array /= t_span

        point_array = np.vstack((x_array, y_array))
        n = point_array.shape[1]+1
        v0, vf = np.array(v0), np.array(vf)
        p0 = point_array[:, 0] + v0/n
        pf = point_array[:,-1] - vf/n

        point_array = np.insert(point_array, 1, values=p0, axis=1)
        point_array = np.insert(point_array,-1, values=pf, axis=1)

        point_array_expand = expand_control_points(point_array)

        self.t0, self.t_span = t0, t_span
        self.x0, self.y0 = x0, y0
        self.p0 = np.array([x0, y0]).reshape(2,1)
        self.point_array = point_array
        self.point_array_expand = point_array_expand
    

    def position(self, time, expand=True):
        time = np.clip(time, self.t0, self.t0+self.t_span)
        t = (time - self.t0) / self.t_span
        p = self.point_array_expand if expand else self.point_array
        position = bezier_curve(t, p, bias=0)
        return position + self.p0
    
    def velocity(self, time, expand=True):
        time = np.clip(time, self.t0, self.t0+self.t_span)
        t = (time - self.t0) / self.t_span
        p = self.point_array_expand if expand else self.point_array
        return bezier_curve(t, p, bias=1)
    
    def acc(self, time, expand=True):
        time = np.clip(time, self.t0, self.t0+self.t_span)
        t = (time - self.t0) / self.t_span
        p = self.point_array_expand if expand else self.point_array
        return bezier_curve(t, p, bias=2)
    
def angle_normal(angle):
    while angle >= np.pi:
        angle -= 2*np.pi
    while angle <= -np.pi:
        angle += 2*np.pi
    return angle

def xy2uv(x, y):
    pixs_per_meter = 200./25.
    u = (200-x*pixs_per_meter).astype(int)
    v = (y*pixs_per_meter+400//2).astype(int)
    #mask = np.where((u >= 0)&(u < 200))[0]
    return u, v

class CostMapDataset(Dataset):
    def __init__(self, data_index, opt, dataset_path='/media/wang/DATASET/CARLA/town01/', evalmode=False):
        self.evalmode = evalmode
        self.data_index = data_index
        self.weights = []
        self.max_dist = opt.max_dist
        self.max_t = opt.max_t
        self.img_step = opt.img_step
        transforms_ = [ transforms.Resize((200, 400), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5)),
            ]
        
        self.transform = transforms.Compose(transforms_)
        self.dataset_path = dataset_path
        self.pose_dict = {}
        self.vel_dict = {}
        self.acc_dict = {}
        self.files_dict = {}
        self.total_len = 0
        self.eval_index = None # eval mode
        self.eval_cnt = 0 # eval mode
        
        for index in self.data_index:
            self.read_pose(index)
            self.read_vel(index)
            self.read_acc(index)
            self.read_img(index)
            self.weights.append(len(self.files_dict[index]))
        
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
        files = glob.glob(self.dataset_path+str(index)+'/ipm/*.png')
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
        
    def __getitem__(self, index):
        while True:
            if self.evalmode:
                if self.eval_index == None:
                    self.eval_index = random.sample(self.data_index,1)[0]
                    self.cnt = 300
                data_index = self.eval_index
                file_name = self.files_dict[data_index][self.cnt]
                self.cnt += 20
                if self.cnt > len(self.files_dict[data_index])-50:
                    self.eval_index = random.sample(self.data_index,1)[0]
                    self.cnt = 300
            else:
                data_index = random.choices(self.data_index, self.weights)[0]
                file_name = random.sample(self.files_dict[data_index][300:-120], 1)[0]
            ts_index = self.files_dict[data_index].index(file_name)
            imgs = []
            try:
                for i in range(-9,1):
                    _file_name = self.files_dict[data_index][ts_index + self.img_step*i]
                    image_path = self.dataset_path + str(data_index)+'/ipm/'+_file_name+'.png'
                    img = Image.open(image_path).convert('L')
                    img = self.transform(img)
                    imgs.append(img)
            except:
                print('get img error:', image_path)
                continue
            imgs = torch.stack(imgs)
            x_0 = self.pose_dict[data_index][file_name][0]
            y_0 = self.pose_dict[data_index][file_name][1]
            yaw = np.deg2rad(self.pose_dict[data_index][file_name][3])
            
            ts_list = []
            relative_t_list = []
            x_list = []
            y_list = []
            vx_list = []
            vy_list = []
            ax_list = []
            ay_list = []
            a_list = []
            collision_flag = False
            collision_x = None
            collision_y = None
            collision_index = None
            for i in range(ts_index, len(self.files_dict[data_index])-100):
                ts = self.files_dict[data_index][i]
                if float(ts)-float(file_name) > self.max_t:
                    break
                else:
                    x_, y_ = self.tf_pose(data_index, ts, yaw, x_0, y_0)
                    u, v = xy2uv(x_, y_)

                    if not collision_flag and u >= 0 and u < 200 and v >=0 and v < 400:
                        if imgs[-1][0][u][v] < -0.3:
                            collision_flag = True
                            collision_x = x_
                            collision_y = y_
                            collision_index = i
                            #break
                    
                    if collision_flag:
                        x_list.append(collision_x)
                        y_list.append(collision_y)
                        vx = 0.
                        vy = 0.
                        a = 0.
                        a_list.append(0.)
                        vx_list.append(0.)
                        vy_list.append(0.)
                        ax_list.append(0.)
                        ay_list.append(0.)
                        ts_list.append(ts)
                        relative_t_list.append(float(ts)-float(file_name))
                    else:
                        x_list.append(x_)
                        y_list.append(y_)
                        vx_ = self.vel_dict[data_index][ts][0]
                        vy_ = self.vel_dict[data_index][ts][1]
                        vx = np.cos(yaw)*vx_ + np.sin(yaw)*vy_
                        vy = np.cos(yaw)*vy_ - np.sin(yaw)*vx_
                        
                        ax_ = self.acc_dict[data_index][ts][0]
                        ay_ = self.acc_dict[data_index][ts][1]
                        ax = ax_*np.cos(yaw) + ay_*np.sin(yaw)
                        ay = ay_*np.cos(yaw) - ax_*np.sin(yaw)
                        ax_list.append(ax)
                        ay_list.append(ay)
                        theta_a = np.arctan2(ay, ax)
                        theta_v = np.arctan2(vy, vx)
                        sign = np.sign(np.cos(theta_a-theta_v))
                        a = sign*np.sqrt(ax*ax + ay*ay)
                        a_list.append(a)
                        vx_list.append(vx)
                        vy_list.append(vy)
                        ts_list.append(ts)
                        relative_t_list.append(float(ts)-float(file_name))
               
            ####################
            
            if collision_flag:
                a_brake = 10
                start_index = collision_index - ts_index
                brake_index = 0
                for i in range(start_index):
                    x_i = x_list[start_index-i]
                    y_i = y_list[start_index-i]
                    safe_dist = np.sqrt((x_i-collision_x)**2+(y_i-collision_y)**2)
                    vx_i = vx_list[start_index-i]
                    vy_i = vy_list[start_index-i]
                    v2 = vx_i**2+vy_i**2
                    brake_dist = v2/(2*a_brake)
                    if brake_dist < safe_dist:
                        brake_index = start_index - i
                        break
                bz_ts = [float(item) for item in ts_list[brake_index:start_index]]
                if len(bz_ts) > 2:
                    bz_x = [x_list[brake_index], collision_x]
                    bz_y = [y_list[brake_index], collision_y]
                    bz_vx = vx_list[brake_index]
                    bz_vy = vy_list[brake_index]
                    #print(bz_ts)
                    bezier = Bezier(bz_ts, bz_x, bz_y, v0=(bz_vx, bz_vy))
                    sample_number = len(bz_ts)
                    time_array = np.linspace(bezier.t0, bezier.t0+bezier.t_span, sample_number)
                    #print(time_array)
                    position_array = bezier.position(time_array, expand=True)
                    velocity_array = bezier.velocity(time_array, expand=True)
                    acc_array = bezier.acc(time_array, expand=True)
                    
                    new_x = position_array[0,:]
                    new_y = position_array[1,:]
                    new_vx = velocity_array[0,:]
                    new_vy = velocity_array[1,:]
                    new_ax = acc_array[0,:]
                    new_ay = acc_array[1,:]
                    for i in range(start_index-brake_index):
                        x_list[brake_index+i] = new_x[i]
                        y_list[brake_index+i] = new_y[i]
                        vx_list[brake_index+i] = new_vx[i]
                        vy_list[brake_index+i] = new_vy[i]
                        ax_list[brake_index+i] = new_ax[i]
                        ay_list[brake_index+i] = new_ay[i]
                        a_list[brake_index+i] = -np.sqrt(new_ax[i]**2+new_ay[i]**2)
                        ts_list[brake_index+i] = str(time_array[i])
                        relative_t_list[brake_index+i] = time_array[i] - float(file_name)
                        #relative_t_list.append(float(ts)-float(file_name))
            
            ####################
            if len(ts_list) == 0:
                continue
            else:
                ts = random.sample(ts_list, 1)[0]
                ts_index = ts_list.index(ts)
                #weights = [np.exp(-0.23*(float(ts)-float(file_name))) for ts in ts_list]
                #sample_ts = random.choices(ts_list, weights)[0]
                #print(weights/sum(weights))
                break
            
        #ts = sample_ts
        # [0 ~ 1]
        t = torch.FloatTensor([float(ts)/self.max_t - float(file_name)/self.max_t])
        # v0
        _vx_0 = self.vel_dict[data_index][file_name][0]
        _vy_0 = self.vel_dict[data_index][file_name][1]
        v_0 = np.sqrt(_vx_0*_vx_0 + _vy_0*_vy_0)
        v_0 = torch.FloatTensor([v_0])
        
        
        x = x_list[ts_index]
        y = y_list[ts_index]
        xy = torch.FloatTensor([x/self.max_dist, y/self.max_dist])# [-1, 1]
        
        # vx, vy
        vx = vx_list[ts_index]
        vy = vy_list[ts_index]
        
        # ax, ay
        ax = ax_list[ts_index]
        ay = ax_list[ts_index]
        
        a = a_list[ts_index]
        a = torch.FloatTensor([a])
        """
        if collision_flag and float(ts) >= float(collision_t):
            ts = collision_t
            # x, y
            x, y = self.tf_pose(data_index, ts, yaw, x_0, y_0)
            xy = torch.FloatTensor([x/self.max_dist, y/self.max_dist])# [-1, 1]
            
            # vx, vy
            vx = 0
            vy = 0
            
            # ax, ay
            ax = 0
            ay = 0
            a = 0
            a = torch.FloatTensor([a])
        else:
            # x, y
            x, y = self.tf_pose(data_index, ts, yaw, x_0, y_0)
            xy = torch.FloatTensor([x/self.max_dist, y/self.max_dist])# [-1, 1]
            
            # vx, vy
            _vx = self.vel_dict[data_index][ts][0]
            _vy = self.vel_dict[data_index][ts][1]
            vx = np.cos(yaw)*_vx + np.sin(yaw)*_vy
            vy = np.cos(yaw)*_vy - np.sin(yaw)*_vx
            
            # ax, ay
            _ax = self.acc_dict[data_index][ts][0]
            _ay = self.acc_dict[data_index][ts][1]
            ax = _ax*np.cos(yaw) + _ay*np.sin(yaw)
            ay = _ay*np.cos(yaw) - _ax*np.sin(yaw)
            
            theta_a = np.arctan2(_ay, _ax)
            theta_v = np.arctan2(_vy, _vx)
            sign = np.sign(np.cos(theta_a-theta_v))
            a = sign*np.sqrt(ax*ax + ay*ay)
            a = torch.FloatTensor([a])
        """
        vxy = torch.FloatTensor([vx, vy])
        axy = torch.FloatTensor([ax, ay])
        x_list = torch.FloatTensor(x_list)
        y_list = torch.FloatTensor(y_list)
        vx_list = torch.FloatTensor(vx_list)
        vy_list = torch.FloatTensor(vy_list)
        a_list = torch.FloatTensor(a_list)
        relative_t_list = torch.FloatTensor(relative_t_list)
        if self.evalmode:
            return {'img': imgs, 't': t, 'xy':xy, 'vxy':vxy, 'axy':axy, 'a':a, 'v_0':v_0,
                    'a_list':a_list,
                    'x_list':x_list, 'y_list':y_list,
                    'vx_list':vx_list, 'vy_list':vy_list,
                    'ts_list':relative_t_list}
        else:
            return {'img': imgs, 't': t, 'xy':xy, 'vxy':vxy, 'axy':axy, 'a':a, 'v_0':v_0}

    def __len__(self):
        return 100000000000
    

class CostMapDataset2(CostMapDataset):
    def __init__(self, data_index, opt, dataset_path='/media/wang/DATASET/CARLA/town01/', evalmode=False):
        self.traj_steps = 8
        self.evalmode = evalmode
        self.data_index = data_index
        self.weights = []
        self.max_dist = opt.max_dist
        self.max_t = opt.max_t
        self.img_step = opt.img_step
        transforms_ = [ transforms.Resize((200, 400), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5)),
            ]
        
        self.transform = transforms.Compose(transforms_)
        self.dataset_path = dataset_path
        self.pose_dict = {}
        self.vel_dict = {}
        self.acc_dict = {}
        self.files_dict = {}
        self.total_len = 0
        self.eval_index = None # eval mode
        self.eval_cnt = 0 # eval mode
        
        for index in self.data_index:
            self.read_pose(index)
            self.read_vel(index)
            self.read_acc(index)
            self.read_img(index)
            self.weights.append(len(self.files_dict[index]))
        
    def __getitem__(self, index):
        while True:
            if self.evalmode:
                if self.eval_index == None:
                    self.eval_index = random.sample(self.data_index,1)[0]
                    self.cnt = 300
                data_index = self.eval_index
                file_name = self.files_dict[data_index][self.cnt]
                self.cnt += 20
                if self.cnt > len(self.files_dict[data_index])-50:
                    self.eval_index = random.sample(self.data_index,1)[0]
                    self.cnt = 300
            else:
                data_index = random.choices(self.data_index, self.weights)[0]
                file_name = random.sample(self.files_dict[data_index][300:-120], 1)[0]
            ts_index = self.files_dict[data_index].index(file_name)
            imgs = []
            try:
                for i in range(-9,1):
                    _file_name = self.files_dict[data_index][ts_index + self.img_step*i]
                    image_path = self.dataset_path + str(data_index)+'/ipm/'+_file_name+'.png'
                    img = Image.open(image_path).convert('L')
                    img = self.transform(img)
                    imgs.append(img)
            except:
                print('get img error:', image_path)
                continue
            imgs = torch.stack(imgs)
            x_0 = self.pose_dict[data_index][file_name][0]
            y_0 = self.pose_dict[data_index][file_name][1]
            yaw = np.deg2rad(self.pose_dict[data_index][file_name][3])
            
            ts_list = []
            relative_t_list = []
            x_list = []
            y_list = []
            vx_list = []
            vy_list = []
            a_list = []
            for i in range(ts_index, len(self.files_dict[data_index])-100):
                ts = self.files_dict[data_index][i]
                if float(ts)-float(file_name) > self.max_t:
                    break
                else:
                    x_, y_ = self.tf_pose(data_index, ts, yaw, x_0, y_0)
                    x_list.append(x_)
                    y_list.append(y_)
                    vx_ = self.vel_dict[data_index][ts][0]
                    vy_ = self.vel_dict[data_index][ts][1]
                    vx = np.cos(yaw)*vx_ + np.sin(yaw)*vy_
                    vy = np.cos(yaw)*vy_ - np.sin(yaw)*vx_
                    
                    ax_ = self.acc_dict[data_index][ts][0]
                    ay_ = self.acc_dict[data_index][ts][1]
                    ax = ax_*np.cos(yaw) + ay_*np.sin(yaw)
                    ay = ay_*np.cos(yaw) - ax_*np.sin(yaw)
                    theta_a = np.arctan2(ay, ax)
                    theta_v = np.arctan2(vy, vx)
                    sign = np.sign(np.cos(theta_a-theta_v))
                    a = sign*np.sqrt(ax*ax + ay*ay)
                    a_list.append(a)
                    vx_list.append(vx)
                    vy_list.append(vy)
                    ts_list.append(ts)
                    relative_t_list.append(float(ts)-float(file_name))
                        
            if len(ts_list) == 0:
                continue
            else:
                #ts = random.sample(ts_list, 1)[0]
                ts_array = random.sample(ts_list, self.traj_steps)
                #weights = [np.exp(-0.23*(float(ts)-float(file_name))) for ts in ts_list]
                #sample_ts = random.choices(ts_list, weights)[0]
                #print(weights/sum(weights))
                break
            
        #ts = sample_ts
        # [0 ~ 1]
        # v0
        _vx_0 = self.vel_dict[data_index][file_name][0]
        _vy_0 = self.vel_dict[data_index][file_name][1]
        v_0 = np.sqrt(_vx_0*_vx_0 + _vy_0*_vy_0)
        v_0 = torch.FloatTensor([v_0])
        v0_array = [v_0]*self.traj_steps

        t_array = []
        xy_array = []
        vxy_array = []
        axy_array = []
        a_array = []
        for ts in ts_array:
            t = torch.FloatTensor([float(ts)/self.max_t - float(file_name)/self.max_t])
            t_array.append(t)
            # x, y
            x, y = self.tf_pose(data_index, ts, yaw, x_0, y_0)
            xy = torch.FloatTensor([x/self.max_dist, y/self.max_dist])# [-1, 1]
            xy_array.append(xy)
            # yaw_t
            #yaw_t = angle_normal(np.deg2rad(self.pose_dict[data_index][ts][3]) - yaw)
            #yaw_t = torch.FloatTensor([yaw_t/np.pi])# [-1, 1]
            
            # vx, vy
            _vx = self.vel_dict[data_index][ts][0]
            _vy = self.vel_dict[data_index][ts][1]
            vx = np.cos(yaw)*_vx + np.sin(yaw)*_vy
            vy = np.cos(yaw)*_vy - np.sin(yaw)*_vx
            vxy_array.append(torch.FloatTensor([vx, vy]))
            
            # ax, ay
            _ax = self.acc_dict[data_index][ts][0]
            _ay = self.acc_dict[data_index][ts][1]
            ax = _ax*np.cos(yaw) + _ay*np.sin(yaw)
            ay = _ay*np.cos(yaw) - _ax*np.sin(yaw)
            axy_array.append(torch.FloatTensor([ax, ay]))
            
            theta_a = np.arctan2(_ay, _ax)
            theta_v = np.arctan2(_vy, _vx)
            sign = np.sign(np.cos(theta_a-theta_v))
            a = sign*np.sqrt(ax*ax + ay*ay)
            a_array.append(a)
        
        t = torch.FloatTensor(t_array)
        v_0 = torch.FloatTensor(v0_array)
        xy = torch.stack(xy_array)
        vxy = torch.stack(vxy_array)
        axy = torch.stack(axy_array)
        a = torch.FloatTensor(a_array)
        
        #vxy = torch.FloatTensor([vx, vy])
        #axy = torch.FloatTensor([ax, ay])
        x_list = torch.FloatTensor(x_list)
        y_list = torch.FloatTensor(y_list)
        vx_list = torch.FloatTensor(vx_list)
        vy_list = torch.FloatTensor(vy_list)
        a_list = torch.FloatTensor(a_list)
        relative_t_list = torch.FloatTensor(relative_t_list)
        
        if self.evalmode:
            return {'img': imgs, 't': t, 'xy':xy, 'vxy':vxy, 'axy':axy, 'a':a, 'v_0':v_0,
                    'a_list':a_list,
                    'x_list':x_list, 'y_list':y_list,
                    'vx_list':vx_list, 'vy_list':vy_list,
                    'ts_list':relative_t_list}
        else:
            return {'img': imgs, 't': t, 'xy':xy, 'vxy':vxy, 'axy':axy, 'a':a, 'v_0':v_0}
    
    
class CARLADataset(Dataset):
    def __init__(self, data_index, dataset_path='/media/wang/DATASET/CARLA/town01/', eval_mode=False):
        self.data_index = data_index
        self.eval_mode = eval_mode
        img_height = 128
        img_width = 256
        
        label_transforms = [
            transforms.Resize((img_height, img_width), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ]
        
        img_transforms = [
            transforms.Resize((img_height, img_width), Image.BICUBIC),
            transforms.ColorJitter(brightness=0.2,contrast=0.2,hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        
        nav_transforms = [
            transforms.Resize((img_height, img_width), Image.BICUBIC),
            transforms.RandomRotation(15, resample=Image.BICUBIC, expand=False),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        
        self.label_transforms = transforms.Compose(label_transforms)
        self.img_transforms = transforms.Compose(img_transforms)
        self.nav_transforms = transforms.Compose(nav_transforms)
        
        self.dataset_path = dataset_path
        self.files_dict = {}
        self.total_len = 0
        
        for index in self.data_index:
            self.read_img(index)
    
    def read_img(self, index):
        files = glob.glob(self.dataset_path+str(index)+'/pm/*.png')
        file_names = []
        for file in files:
            file_name = file.split('/')[-1][:-4]
            file_names.append(file_name)
        file_names.sort()
        self.files_dict[index] = file_names
        
    def __getitem__(self, index):
        mirror = False#True if random.random() > 0.5 else False
        data_index = random.sample(self.data_index, 1)[0]
        while True:
            try:
                file_name = random.sample(self.files_dict[data_index], 1)[0]
                # img
                img_path = self.dataset_path + str(data_index)+'/img/'+file_name+'.png'
                img = Image.open(img_path).convert("RGB")
                # nav
                nav_path = self.dataset_path + str(data_index)+'/nav/'+file_name+'.png'
                nav = Image.open(nav_path).convert("RGB")
                # label
                label_path = self.dataset_path + str(data_index)+'/pm/'+file_name+'.png'
                label = Image.open(label_path).convert('L')
                
                # mirror the inputs
                if mirror:
                    img = Image.fromarray(np.array(img)[:, ::-1, :], 'RGB')
                    nav = Image.fromarray(np.array(nav)[:, ::-1, :], 'RGB')
                    label = Image.fromarray(np.array(label)[:, ::-1], 'L')
                
                img = self.img_transforms(img)
                nav = self.nav_transforms(nav)
                label = self.label_transforms(label)
                break
            except:
                pass
        if not self.eval_mode:
            input_img = torch.cat((img, nav), 0)
            return {'A': input_img, 'B': label}
        else:
            return {'A1': img, 'A2': nav, 'B': label, 'file_name':file_name}

    def __len__(self):
        return 100000000000
    
class FakeCostMapDataset(Dataset):
    def __init__(self, data_index, opt, dataset_path='/media/wang/DATASET/CARLA_HUMAN/town01/', evalmode=False):
        self.evalmode = evalmode
        self.data_index = data_index
        self.max_dist = opt.max_dist
        self.max_t = opt.max_t
        img_height = 200
        img_width = 400
        img_transforms = [
            transforms.Resize((img_height, img_width), Image.BICUBIC),
            transforms.ColorJitter(brightness=0.2,contrast=0.2,hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        
        self.transform = transforms.Compose(img_transforms)
        
        nav_transforms = [
            transforms.Resize((img_height, img_width), Image.BICUBIC),
            transforms.RandomRotation(15, resample=Image.BICUBIC, expand=False),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        
        self.nav_transforms = transforms.Compose(nav_transforms)
        
        self.dataset_path = dataset_path
        self.pose_dict = {}
        self.vel_dict = {}
        self.files_dict = {}
        self.total_len = 0
        
        for index in self.data_index:
            self.read_pose(index)
            self.read_vel(index)
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
    
    def read_img(self, index):
        files = glob.glob(self.dataset_path+str(index)+'/ipm/*.png')
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
        
    def __getitem__(self, index):
        data_index = random.sample(self.data_index, 1)[0]
        while True:
            file_name = random.sample(self.files_dict[data_index][:-120], 1)[0]
            image_path = self.dataset_path + str(data_index)+'/img/'+file_name+'.png'
            img = Image.open(image_path).convert("RGB")
            img = self.transform(img)
            # nav
            nav_path = self.dataset_path + str(data_index)+'/nav/'+file_name+'.png'
            nav = Image.open(nav_path).convert("RGB")
            nav = self.nav_transforms(nav)
            
            input_img = torch.cat((img, nav), 0)
            
            x_0 = self.pose_dict[data_index][file_name][0]
            y_0 = self.pose_dict[data_index][file_name][1]
            yaw = np.deg2rad(self.pose_dict[data_index][file_name][3])
            
            self.files_dict[data_index].sort()
            ts_index = self.files_dict[data_index].index(file_name)
            ts_list = []
            x_list = []
            y_list = []
            for i in range(ts_index+1, len(self.files_dict[data_index])-100):
                ts = self.files_dict[data_index][i]
                
                _x_t = self.pose_dict[data_index][ts][0]
                _y_t = self.pose_dict[data_index][ts][1]
                distance = np.sqrt((x_0-_x_t)**2+(y_0-_y_t)**2)
                if distance > self.max_dist or (float(ts)-float(file_name) > self.max_t):
                    break
                else:
                    if distance < 0.03:
                        pass
                    else:
                        x_, y_ = self.tf_pose(data_index, ts, yaw, x_0, y_0)
                        x_list.append(x_)
                        y_list.append(y_)
                        ts_list.append(ts)
                        
            if len(ts_list) == 0:
                continue
            else:
                ts = random.sample(ts_list, 1)[0]
                break
        # [0 ~ 1]
        t = torch.FloatTensor([float(ts)/self.max_t - float(file_name)/self.max_t])
        _vx_0 = self.vel_dict[data_index][file_name][0]
        _vy_0 = self.vel_dict[data_index][file_name][1]
        vx_0 = np.cos(yaw)*_vx_0 + np.sin(yaw)*_vy_0
        #vy_0 = np.cos(yaw)*_vy_0 - np.sin(yaw)*_vx_0
        v_0 = torch.FloatTensor([vx_0])
        x, y = self.tf_pose(data_index, ts, yaw, x_0, y_0)
        # [-1, 1]
        xy = torch.FloatTensor([x/self.max_dist, y/self.max_dist])
        
        yaw_t = angle_normal(np.deg2rad(self.pose_dict[data_index][ts][3]) - yaw)
        # [-1, 1]
        yaw_t = torch.FloatTensor([yaw_t/np.pi])
    
        _vx = self.vel_dict[data_index][ts][0]
        _vy = self.vel_dict[data_index][ts][1]
        vx = np.cos(yaw)*_vx + np.sin(yaw)*_vy
        vy = np.cos(yaw)*_vy - np.sin(yaw)*_vx
        
        vxy = torch.FloatTensor([vx, vy])
        x_list = torch.FloatTensor(x_list)
        y_list = torch.FloatTensor(y_list)
        if self.evalmode:
            return {'img': input_img, 't': t, 'xy':xy, 'vxy':vxy, 'v_0':v_0, 'yaw_t': yaw_t, 'x_list':x_list, 'y_list':y_list}
        else:
            return {'img': input_img, 't': t, 'xy':xy, 'vxy':vxy, 'v_0':v_0}

    def __len__(self):
        return 100000000000
    
if __name__ == '__main__':
    import argparse
    from datetime import datetime
    from PIL import Image, ImageDraw
    from torch.utils.data import DataLoader
    random.seed(datetime.now())
    torch.manual_seed(666)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default="mu-log_var-test", help='name of the dataset')
    parser.add_argument('--width', type=int, default=400, help='image width')
    parser.add_argument('--height', type=int, default=200, help='image height')
    parser.add_argument('--scale', type=float, default=25., help='longitudinal length')
    parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
    parser.add_argument('--img_step', type=int, default=3, help='RNN input image step')
    parser.add_argument('--traj_steps', type=int, default=8, help='traj steps')
    parser.add_argument('--max_dist', type=float, default=25., help='max distance')
    parser.add_argument('--max_t', type=float, default=3., help='max time')
    opt = parser.parse_args()
    
    train_loader = DataLoader(CostMapDataset(data_index=[1,2,3,4,5,6,7,9,10], opt=opt, dataset_path='/media/wang/DATASET/CARLA_HUMAN/town01/', evalmode=True), batch_size=1, shuffle=False)
    
    cnt = 0
    for i, batch in enumerate(train_loader):
        img = batch['img'][:,-1,:].clone().data.numpy().squeeze()*127+128
        img = Image.fromarray(img).convert("RGB")
        draw =ImageDraw.Draw(img)
        
    
        real_x = batch['x_list'].squeeze().data.numpy()
        real_y = batch['y_list'].squeeze().data.numpy()
        real_u, real_v = xy2uv(real_x, real_y)
    
        for i in range(len(real_u)-1):
            draw.line((real_v[i], real_u[i], real_v[i]+1, real_u[i]+1), 'blue')
            draw.line((real_v[i], real_u[i], real_v[i]-1, real_u[i]-1), 'blue')
            #draw.line((real_v[i]+1, real_u[i], real_v[i+1]+1, real_u[i+1]), 'blue')
            #draw.line((real_v[i]-1, real_u[i], real_v[i+1]-1, real_u[i+1]), 'blue')
        
        #if cnt % 10 == 0:
        #    img.show()
        cnt += 1
        if cnt > 50:
            break
        #break
    
    
    
    