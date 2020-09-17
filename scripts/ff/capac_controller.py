import carla_utils as cu
import time
import numpy as np
import carla


class Param(object):
    def __init__(self):
        '''vehicle'''
        self.L = 2.405
        self.max_acceleration = 1.0
        self.min_acceleration = -1.0
        self.max_steer = 1.0

        '''PID'''
        self.v_pid_Kp = 1.00#1.00
        self.v_pid_Ki = 0.00
        self.v_pid_Kd = 0.00

        '''RWPF'''
        self.k_k = 1.235
        self.k_theta = 0.456
        self.k_e = 0.11#0.1386

        '''debug'''
        self.curvature_factor = 1.0


class CapacController(object):
    def __init__(self, world, vehicle, frequency):
        '''parameter'''
        config = Param()
        self.world = world

        self.vehicle = vehicle

        self.L = config.L
        self.max_steer = config.max_steer
        self.dt = 1. / frequency

        '''debug'''
        self.curvature_factor = config.curvature_factor

        '''PID'''
        self.Kp, self.Ki, self.Kd = config.v_pid_Kp, config.v_pid_Ki, config.v_pid_Kd
        self.max_a = config.max_acceleration
        self.min_a = config.min_acceleration
        self.last_error = 0
        self.sum_error = 0

        '''RWPF'''
        self.k_k, self.k_theta, self.k_e = config.k_k, config.k_theta, config.k_e
    

    def run_step(self, trajectory, index, state0):
        '''
        Args:
            trajectory = {'time':plan_time, 'x':x, 'y':y, 'vx':vx, 'vy':vy, 'ax':ax, 'ay':ay, 'a':a}
        '''
        #print(trajectory)
        time_stamp = time.time()
        current_state = cu.getActorState('odom', time_stamp, self.vehicle)
        current_state = current_state.world_to_local_2D(state0, 'base_link')

        x, y, vx, vy = trajectory['x'][index], trajectory['y'][index], trajectory['vx'][index],  trajectory['vy'][index]
        ax, ay = trajectory['ax'][index], trajectory['ay'][index]
        a, t = trajectory['a'][index], trajectory['time']
        theta, v = np.arctan2(vy, vx), np.hypot(vx, vy)
        k = (vx*ay-vy*ax)/(v**3)
        target_state = cu.State('base_link', time_stamp, x=x, y=y, theta=theta, v=v, a=a, k=k, t=t)
        target_state.y = target_state.y-0.2
        
        steer = self.rwpf(current_state, target_state)
        #org_tv = target_state.v
        target_state.v = target_state.v/(1+0.6*abs(steer))
        throttle, brake = self.pid(current_state, target_state)
        target_state.k = k * self.curvature_factor
        
        # debug
        global_target = target_state.local_to_world_2D(state0, 'odom')
        localtion = carla.Location(x = global_target.x, y=global_target.y, z=2.0)
        self.world.debug.draw_point(localtion, size=0.2, color=carla.Color(255,0,0), life_time=5.0)
        # throttle, brake = 1, 0
        throttle += 0.7
        throttle = np.clip(throttle, 0., 1.)
        #throttle = throttle/(1+abs(steer))
        
        #if throttle > 0 and abs(global_vel) < 0.8 and abs(v_r) < 1.0:
        if throttle > 0 and abs(current_state.v) < 1.0 and abs(target_state.v) < 1.0:
            throttle = 0.
            brake = 1.
            steer = 0.

        return carla.VehicleControl(throttle=throttle, brake=brake, steer=steer)
    

    def pid(self, current_state, target_state):
        v_current = current_state.v
        v_target = target_state.v
        error = v_target - v_current
        
        acceleration = self.Kp * error
        acceleration += self.Ki * self.sum_error * self.dt
        acceleration += self.Kd * (error - self.last_error) / self.dt

        self.last_error = error
        self.sum_error += error
        '''eliminate drift'''
        if abs(self.sum_error) > 10:
            self.sum_error = 0.0

        throttle = np.clip(acceleration, 0, self.max_a)
        brake = -np.clip(acceleration, self.min_a, 0)
        
        # print(v_target, v_current, '    ', throttle, brake)
        return throttle, brake


    def rwpf(self, current_state, target_state):
        xr = target_state.x
        yr = target_state.y
        thetar = target_state.theta
        vr = target_state.v
        kr = target_state.k

        dx = current_state.x - xr
        dy = current_state.y - yr
        tx = np.cos(thetar)
        ty = np.sin(thetar)
        e = dx*ty - dy*tx
        theta_e = cu.basic.pi2pi(current_state.theta - thetar)

        alpha = 1.8
        
        w1 = self.k_k * vr*kr*np.cos(theta_e)
        w2 = - self.k_theta * np.fabs(vr)*theta_e
        w3 = (self.k_e*vr*np.exp(-theta_e**2/alpha))*e
        w = (w1+w2+w3)*0.8
        #print(dx, dy, current_state.theta, xr, yr, thetar)
        if current_state.v < 0.02:
            steer = 0
        else:
            steer = np.arctan2(w*self.L, current_state.v) * 2 / np.pi * self.max_steer
        
        #print(w, steer)

        return steer




if __name__ == "__main__":
    client, world, town_map = cu.connect_to_server('127.0.0.1', 2000, timeout=2.0, map_name='None')
    vehicle = cu.get_actor(world, 'vehicle.bh.crossbike', 'hero')


    c = CapacController(vehicle, 10)
    
    trajectory = {'time':1, 'x':1, 'y':1, 'vx':1, 'vy':1, 'ax':1, 'ay':1, 'a':1}

    while True:
        a = c.run_step(trajectory)
        print(a)
        time.sleep(0.1)