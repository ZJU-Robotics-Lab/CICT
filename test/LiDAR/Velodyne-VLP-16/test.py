import time
import struct
import socket
import numpy as np

from vtk_visualize import vtk_visualize

#HOST = '10.12.218.167'
PORT = 2368

NUM_LASERS = 16
LASER_ANGLES = [-15, 1, -13, 3, -11, 5, -9, 7, -7, 9, -5, 11, -3, 13, -1, 15]
DISTANCE_RESOLUTION = 0.002
ROTATION_MAX_UNITS = 36000

soc = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
soc.bind(('', PORT))

def calc(dis, azimuth, laser_id, timestamp):
    R = dis * DISTANCE_RESOLUTION
    omega = LASER_ANGLES[laser_id] * np.pi / 180.0
    alpha = azimuth / 100.0 * np.pi / 180.0
    X = R * np.cos(omega) * np.sin(alpha)
    Y = R * np.cos(omega) * np.cos(alpha)
    Z = R * np.sin(omega)
    return [X, Y, Z]

if __name__ == '__main__':
    points = []
    prev_theta = None
    scan_index = 0 # cicle num
    theta = 0
    start = time.time()
    for _ in range(90):
        data = soc.recv(2000)
        if len(data) > 0:
            assert len(data) == 1206
            
            # main package
            timestamp, factory = struct.unpack_from("<IH", data, offset=1200)
            assert factory == 0x2237, hex(factory)  # 0x22=VLP-16, 0x37=Strongest Return

            seq_index = 0
            for offset in range(0, 1200, 100):
                # 12 bags' head
                flag, theta = struct.unpack_from("<HH", data, offset)
                print('theta:', theta)
                assert flag == 0xEEFF
                
                # 2*16 data
                for step in range(2):
                    seq_index += 1
                    theta += step
                    theta %= ROTATION_MAX_UNITS
                    if prev_theta is not None and theta < prev_theta:
                        # one cicle
                        scan_index += 1

                    prev_theta = theta
                    # H-distance (2mm step), B-reflectivity
                    arr = struct.unpack_from('<' + "HB" * 16, data, offset + 4 + step * 48)
                    for i in range(NUM_LASERS):
                        time_offset = (55.296 * seq_index + 2.304 * i) / 1000000.0
                        if arr[i * 2] != 0:
                            points.append(calc(arr[i * 2], theta, i, timestamp + time_offset))

    end = time.time()
    print(round(end - start, 4), 's')
    vtk_visualize(points)