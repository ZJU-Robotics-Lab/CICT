import sys
import struct
import socket
import threading
import numpy as np
from queue import Queue
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtGui
try:
    from pyquery import PyQuery as pq
except:
    pass
     
USE_CUDA = False
if USE_CUDA:
    import torch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


HOST = '10.12.218.255'
IP = '10.12.218.167'
GATEWAY = '10.12.218.1'
RPM = '600'

class LiDAR:
    def __init__(self, port=2368):
        self.PORT = port
        self.NUM_LASERS = 16
        self.LASER_ANGLES = [-15, 1, -13, 3, -11, 5, -9, 7, -7, 9, -5, 11, -3, 13, -1, 15]
        self.DISTANCE_RESOLUTION = 0.002
        self.ROTATION_MAX_UNITS = 36000
        
        self.soc = None
        self.data_queue = Queue(1000)
        
        self.stop_read_event = threading.Event()
        
        self.read_cyclic = threading.Thread(
            target=self.read_data, args=()
        )
        
        self.points = []
        self.scan_index = 0 # cicle num
        self.theta = 0
        self.list_dis = []
        self.list_theta = []
        self.list_id = []
        self.set_param()
    
    def set_param(self, host = HOST, ip = IP, gateway = GATEWAY, rpm = RPM):
        try:
            pq('http://10.12.218.167/cgi/setting/host', {'addr':HOST }, method='post', verify=True)
        except:
            pass
        try:
            pq('http://10.12.218.167/cgi/setting', {'rpm':RPM }, method='post',verify=True)
        except:
            pass
        try:
            pq('http://10.12.218.167/cgi/setting/net', {'addr':IP, 'gateway' : GATEWAY}, method='post',verify=True)
        except:
            pass

    def start(self):
        self.soc = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.soc.bind(('', self.PORT))
        self.stop_read_event.clear()
        self.read_cyclic.start()
        
    def close(self):
        self.stop_read_event.set()
        if self.soc is not None:
            self.soc.close()
            
    def get(self):
        if not self.data_queue.empty():
            return self.data_queue.get()
        else:
            return None
        
    def clear(self):
        self.data_queue.queue.clear()
        
    def calc(self, dis, azimuth, laser_id):
        dis = np.array(dis)
        azimuth = np.array(azimuth)
        R = dis * self.DISTANCE_RESOLUTION
        omega = [self.LASER_ANGLES[item] * np.pi / 180.0 for item in laser_id]
        alpha = azimuth / 100.0 * np.pi / 180.0
        X = R * np.cos(omega) * np.sin(alpha)
        Y = R * np.cos(omega) * np.cos(alpha)
        Z = R * np.sin(omega)
        _pts = np.dstack((X,Y))
        pts = np.dstack((_pts,Z))[0]
        return pts
    
    def calc_cuda(self, dis, azimuth, laser_id):
        dis = np.array(dis)
        dis = torch.from_numpy(dis).to(device)
        
        azimuth = np.array(azimuth)
        azimuth = torch.from_numpy(azimuth).to(device)
        
        R = dis * self.DISTANCE_RESOLUTION
        
        omega = [self.LASER_ANGLES[item] * np.pi / 180.0 for item in laser_id]
        omega = np.array(omega)
        omega = torch.from_numpy(omega).to(device)
        
        alpha = azimuth / 100.0 * np.pi / 180.0
        X = R * torch.cos(omega) * torch.sin(alpha)
        Y = R * torch.cos(omega) * torch.cos(alpha)
        Z = R * torch.sin(omega)
        
        _pts = torch.stack((X, Y), 1)
        pts = torch.cat((_pts, Z.unsqueeze(-1)), 1)
        return pts.cpu().numpy()
    
    def read_data(self):
        while not self.stop_read_event.is_set():
            data = self.soc.recv(2000)
            if len(data) > 0:
                assert len(data) == 1206
                
                # main package
                timestamp, factory = struct.unpack_from("<IH", data, offset=1200)
                assert factory == 0x2237, hex(factory)  # 0x22=VLP-16, 0x37=Strongest Return
                seq_index = 0
                for offset in range(0, 1200, 100):
                    # 12 bags' head
                    flag, theta = struct.unpack_from("<HH", data, offset)
                    assert flag == 0xEEFF
                    # 2*16 data
                    for step in range(2):
                        seq_index += 1
                        theta += step*20
                        if theta > self.ROTATION_MAX_UNITS:
                            theta %= self.ROTATION_MAX_UNITS
                            # one cicle finish
                            self.scan_index += 1
                            if self.data_queue.full():
                                print('WARNNING: LiDAR data queue is FULL !')
                                with self.data_queue.mutex:
                                    self.data_queue.queue.clear()
                            
                            if USE_CUDA:
                                self.points = self.calc_cuda(self.list_dis, self.list_theta, self.list_id)
                            else:
                                self.points = self.calc(self.list_dis, self.list_theta, self.list_id)
                            self.data_queue.put(self.points)
                            self.list_dis = []
                            self.list_theta = []
                            self.list_id = []
                            self.points = []

                        # H-distance (2mm step), B-reflectivity
                        arr = struct.unpack_from('<' + "HB" * 16, data, offset + 4 + step * 48)
                        for i in range(self.NUM_LASERS):
                            #time_offset = (55.296 * seq_index + 2.304 * i) / 1000000.0
                            #ts = timestamp + time_offset
                            if arr[i * 2] != 0:
                                #self.points.append(self.calc(arr[i * 2], theta, i))
                                self.list_dis.append(arr[i * 2])
                                self.list_theta.append(theta)
                                self.list_id.append(i)

            #self.points.append(self.calc(self.list_dis, self.list_theta, self.list_id))

            
            
class Visualizer(object):
    def __init__(self, data_queue):
        self.data_queue = data_queue
        self.traces = dict()
        self.app = QtGui.QApplication(sys.argv)
        self.w = gl.GLViewWidget()
        self.w.opts['distance'] = 40
        self.w.setWindowTitle('Visualizer')
        self.w.setGeometry(0, 110, 1920, 1080)
        self.w.show()

        # create the background grids
        gx = gl.GLGridItem()
        gx.rotate(90, 0, 1, 0)
        gx.translate(-10, 0, 10)
        self.w.addItem(gx)
        gy = gl.GLGridItem()
        gy.rotate(90, 1, 0, 0)
        gy.translate(0, -10, 10)
        self.w.addItem(gy)
        gz = gl.GLGridItem()
        gz.translate(0, 0, 0)
        self.w.addItem(gz)

        pts = np.array([[0,0,0]])
        self.traces[0] = gl.GLScatterPlotItem(pos=pts, color=(1.,1.,1.,0.), size=0.1)
        self.w.addItem(self.traces[0])
        
        self.points = pts

    def start(self):
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()
            
    def close(self):
        QtGui.QApplication.instance().quit()

    def set_plotdata(self, points, color):
        self.traces[0].setData(pos=points, color=color)

    def update(self):
        if not self.data_queue.empty():
            pts = self.data_queue.get()
            self.points = pts

        self.set_plotdata(
            points=self.points,
            color=(0.,1.,1.,1.)#self.get_color(self.points)
        )

    def animation(self):
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(16)
        self.start()
        
    def get_color(self, pts):
        z_max = np.max(pts, axis=0)[2]
        z_min = np.min(pts, axis=0)[2]
        z_avg = np.mean(pts, axis=0)[2]
        delta = min(z_max - z_avg, z_avg - z_min)
        z_max = z_avg + delta
        z_min = z_avg - delta
        
        colors = np.ones((pts.shape[0], 4))
        for i in range(len(pts)):
            color = (pts[i][2] - z_min)/(z_max - z_min)
            color = max(0, min(color, 1))
            colors[i][0] = 2*color-1 if color > 0.5 else 0
            colors[i][1] = 2 - 2*color if color > 0.5 else 2*color
            colors[i][2] = 0 if color > 0.5 else 1 - 2*color
        return colors
                                
if __name__ == '__main__':
    lidar = LiDAR()
    lidar.start()
    v = Visualizer(lidar.data_queue)
    # it will block here and not stop
    v.animation()
    v.close()
    lidar.close()