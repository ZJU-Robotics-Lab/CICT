import carla_utils as cu
import time
import numpy as np
import carla


class Param(object):
    def __init__(self):
        '''vehicle'''
        self.L = 2.875
        self.max_acceleration = 1.0
        self.min_acceleration = -1.0
        self.max_steer = 1.0

        '''PID'''
        self.v_pid_Kp = 1.00
        self.v_pid_Ki = 0.00
        self.v_pid_Kd = 0.05

        '''RWPF'''
        self.k_k = 1.235
        self.k_theta = 0.456
        self.k_e = 0.1386

        '''debug'''
        self.curvature_factor = 1.0


class CapacController(object):
    def __init__(self, vehicle, frequency):
        '''parameter'''
        config = Param()

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
    

    def run_step(self, trajectory):
        '''
        Args:
            trajectory = {'time':plan_time, 'x':x, 'y':y, 'vx':vx, 'vy':vy, 'ax':ax, 'ay':ay, 'a':a}
        '''
        time_stamp = time.time()
        current_state = cu.getActorState('odom', time_stamp, self.vehicle)

        x, y, vx, vy = trajectory['x'], trajectory['y'], trajectory['vx'],  trajectory['vy']
        ax, ay = trajectory['ax'], trajectory['ay']
        a, t = trajectory['a'], trajectory['time']
        theta, v = np.arctan2(vy, vx), np.hypot(vx, vy)
        k = (vx*ay-vy*ax)/(v**3)
        target_state = cu.State('odom', time_stamp, x=x, y=y, theta=theta, v=v, a=a, k=k, t=t)

        throttle, brake = self.pid(current_state, target_state)
        target_state.k = k * self.curvature_factor
        steer = self.rwpf(current_state, target_state)

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
        w = w1+w2+w3

        if current_state.v < 1.001:
            steer = 0
        else:
            steer = np.arctan2(w*self.L, current_state.v) * 2 / np.pi * self.max_steer

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