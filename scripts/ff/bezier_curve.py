
import numpy as np
import time
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


'''
Args:
    t: [0,1]
'''

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
    def __init__(self, time_list, x_list, y_list, v0, vf=(-0.0001,-0.0001)):
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







import matplotlib.pyplot as plt

if __name__ == "__main__":
    t = [1547123737374427, 1547123737394436, 1547123737414446, 1547123737434455, 1547123737454465, 1547123737474474]
    t = list(np.array(t) / 1000000)
    
    x = [-170.17798699997365, -169.96834699995816, -169.71550199948251, -169.50583699997514, -169.29630899988115, -169.08685099985451]
    y = [180.99746099999174, 180.98115799995139, 181.16259399999399, 181.14671300002374, 181.13088099996094, 181.11514000000898]

    bezier = Bezier(t, x, y, v0=(2., 7.))

    sample_number = 60
    time_array = np.linspace(bezier.t0, bezier.t0+bezier.t_span, sample_number)
    position_array = bezier.position(time_array, expand=True)
    velocity_array = bezier.velocity(time_array, expand=True)

    plt.subplots(1)
    plt.plot(bezier.point_array[0,:]+bezier.x0, bezier.point_array[1,:]+bezier.y0, 'or')
    plt.plot(position_array[0,:], position_array[1,:], '-b')
    plt.gca().set_aspect('equal', adjustable='box')

    plt.subplots(1)
    plt.plot(time_array, velocity_array[0,:], 'or')
    plt.subplots(1)
    plt.plot(time_array, velocity_array[1,:], 'or')

    plt.show()
