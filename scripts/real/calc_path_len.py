
import numpy as np
import matplotlib.pyplot as plt

def calc_len(index):
    file_name = '/home/wang/video/data'+str(index)+'/gps/gps.txt'
    with open(file_name, 'r') as file:
        lines = file.readlines()
        ts = []
        xs = []
        ys = []
        yaws = []
        for line in lines:
            sp_line = line.split()
            t = float(sp_line[0])
            x = float(sp_line[1])
            y = float(sp_line[2])
            yaw = float(sp_line[3])
            ts.append(t)
            xs.append(x)
            ys.append(y)
            yaws.append(yaw)
        
        xs = np.array(xs)
        ys = np.array(ys)

        xs = xs - xs[1]
        ys = ys - ys[1]
        mask = np.where(abs(xs) < 500)[0]
        xs = xs[mask]
        ys = ys[mask]
        mask = np.where(abs(ys) < 500)[0]
        xs = xs[mask]
        ys = ys[mask]

        
        dists = np.hypot(xs[:-1]-xs[1:], ys[:-1]-ys[1:])
        mask = np.where(dists < 0.9)[0]
        xs = xs[mask]
        ys = ys[mask]
        dists = dists[mask]
        dist = np.sum(dists)
        print(dist)
        plt.figure(figsize=(5, 5))

        plt.plot(xs, ys, color='red', label='Ours')

        # plt.title('Success Rate / %')
        # plt.xlabel('Delay Time / ms')
        # plt.ylabel('Success Rate / %')
        # plt.xlim(0, 1100)
        # plt.ylim(0, 100)
        plt.legend()
        plt.tight_layout()
        # plt.savefig('delay.png')
        plt.show()


for i in range(2,6):
    calc_len(i)