#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import numpy as np
import socket
import threading

import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtGui

class Visualizer(object):
    def __init__(self):
        self.point_size = 3.0
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
        self.traces[0] = gl.GLScatterPlotItem(pos=pts, color=(1.,1.,1.,0.), size=self.point_size)
        self.w.addItem(self.traces[0])
        
        self.points = pts
        
        self.PORT = 6666
        self.stop_read_event = threading.Event()
        
        self.read_cyclic = threading.Thread(
            target=self.read_data, args=()
        )
        
    def start(self):
        self.soc = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.soc.bind(('', self.PORT))
        self.stop_read_event.clear()
        self.read_cyclic.start()
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()
            
    def close(self):
        self.stop_read_event.set()
        if self.soc is not None:
            self.soc.close()
        QtGui.QApplication.instance().quit()

    def read_data(self):
        while not self.stop_read_event.is_set():
            data = self.soc.recv(65535)
            self.points = np.frombuffer(data,dtype="float32").reshape(int(len(data)/12), 3)
        
    def set_plotdata(self, points, color):
        self.traces[0].setData(pos=points, color=color, size=self.point_size)

    def update(self):
        #self.points = global_pc.T
        self.set_plotdata(
            points=self.points,
            color=(0.,1.,1.,1.)
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
    v = Visualizer()
    v.animation()
    v.close()