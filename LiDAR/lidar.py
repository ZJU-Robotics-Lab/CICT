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

import velodyne

HOST = '10.12.218.255'
IP = '10.12.218.167'
GATEWAY = '10.12.218.1'
RPM = '600'

class LiDAR:
    def __init__(self, port=2368):
        self.PORT = port
        self.soc = None
        self.data_queue = Queue(1000)
        
        self.stop_read_event = threading.Event()
        
        self.read_cyclic = threading.Thread(
            target=self.read_data, args=()
        )
        
        self.points = np.zeros((384*720, 4))
        self.scan_index = 0 # cicle num

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
        """
        if not self.data_queue.empty():
            return self.data_queue.get()
        else:
            return None
        """
        return self.points.T
        
    def clear(self):
        self.data_queue.queue.clear()
        
    def read_data(self):
        while not self.stop_read_event.is_set():
            data = self.soc.recv(2000)
            if len(data) > 0:
                assert len(data) == 1206
                
                # main package
                timestamp, factory = struct.unpack_from("<IH", data, offset=1200)
                assert factory == 0x2237, hex(factory)  # 0x22=VLP-16, 0x37=Strongest Return
                result = velodyne.parse_data(data[:1200])
                #intensity = result[:,3]
                #mask = np.where((intensity > 20))[0]
                #result = result[:,:3]
                #result = result[mask,:]
                self.points[384*self.scan_index:384*(self.scan_index+1), :] = result#[:,:3]
                self.scan_index += 1
                if self.scan_index >= 720:
                    self.scan_index = 0
                    #self.data_queue.put(self.points)
                    #self.points = None
     
            
class Visualizer(object):
    def __init__(self, lidar):
        self.lidar = lidar
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
        #if not self.data_queue.empty():
        #    pts = self.data_queue.get()
        #    self.points = pts
        self.points = self.lidar.get().T

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
    v = Visualizer(lidar)
    # it will block here and not stop
    v.animation()
    v.close()
    lidar.close()
