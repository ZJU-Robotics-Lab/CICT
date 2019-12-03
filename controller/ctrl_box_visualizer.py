#!/usr/bin/env python3

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import time
import threading
import sys
import pyqtgraph.parametertree.parameterTypes as pTypes
from pyqtgraph.parametertree import Parameter, ParameterTree, ParameterItem, registerParameterType
from controller import Controller

class CtrlBoxVisualizer(object):
    def __init__(self, controller):
        self.ctrl = controller
        self.items = ['Cmd Speed',    # 0
                'Cmd Rotation',       # 1
                'Cmd AccTime',        # 2
                'Cmd Direction',      # 3
                'Max Speed',          # 4
                'Max Rotation',       # 5
                'Speed PWM',          # 6
                'Rotation',           # 7
                'Speed km/h',         # 8
                'Rotation Error',     # 9
                'Motor Error',        # 10
                'Battery Temperature',# 11
                'Battery Power',      # 12
                'Motor Current',      # 13
                'Auto Control',       # 14
                'Emergenry Stop'      # 15
                ]
        self.find_function = {
            self.items[0] : self.ctrl.get_cmd_speed(),
            self.items[1] : self.ctrl.get_cmd_rotation(),
            self.items[2] : self.ctrl.get_cmd_acc_time(),
            self.items[3] : self.judge_dir(),
            self.items[4] : self.ctrl.get_max_speed(),
            self.items[5] : self.ctrl.get_max_rotation(),
            self.items[6] : self.ctrl.get_cur_motor_pwm_speed(),
            self.items[7] : self.ctrl.get_cur_rotation(),
            self.items[8] : self.ctrl.get_cur_speed(),
            self.items[9] : self.ctrl.get_cur_rot_error(),
            self.items[10] : self.ctrl.get_cur_ctr_error(),
            self.items[11] : self.ctrl.get_cur_battery_temperature(),
            self.items[12] : self.ctrl.get_cur_battery_power(),
            self.items[13] : self.ctrl.get_cur_motor_current(),
            self.items[14] : self.ctrl.get_cur_ctr_auto(),
            self.items[15] : self.ctrl.get_cur_ctr_emergenry_stop(),
        }
        self.params_children = []
        self.get_new_param_children()
        self.app = QtGui.QApplication(sys.argv)
        self.params = Parameter.create(name='params', type='group', children=self.params_children)
        self.params_tree = ParameterTree()
        self.params_tree.setParameters(self.params, showTop=False)
        self.params_tree.setWindowTitle('pyqtgraph example: Parameter Tree')
        self.win = QtGui.QWidget()
        self.layout = QtGui.QGridLayout()
        self.win.setLayout(self.layout)
        self.layout.addWidget(QtGui.QLabel("The controller params"), 0,  0, 1, 2)
        self.layout.addWidget(self.params_tree, 1, 0, 1, 1)
        self.win.show()
        self.win.resize(800,800)

    def judge_dir(self):
        if self.ctrl.get_cmd_forward():
            return 'forward'
        elif self.ctrl.get_cmd_backward():
            return 'backward'
        else:
            return 'stop'

    def start(self):
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()

    def stop(self):
        QtGui.QApplication.instance().quit()

    def update(self):
        self.get_new_param_children()
        self.params.clearChildren()
        self.params.addChildren(self.params_children)
        self.params_tree.setParameters(self.params, showTop=False)
        self.layout.addWidget(self.params_tree, 1, 0, 1, 1)

    def get_new_param_children(self):
        self.params_children = [
            {'name': 'Command Params', 'type': 'group', 'children': [
                {'name': self.items[0], 'type': 'int', 'value': self.find_function[self.items[0]], 'siPrefix': True, 'suffix': 'rpm', 'readonly': True},
                {'name': self.items[1], 'type': 'int', 'value': self.find_function[self.items[1]], 'readonly': True},
                {'name': self.items[2], 'type': 'int', 'value': self.find_function[self.items[2]], 'siPrefix': False, 'suffix': 's', 'readonly': True},
                {'name': self.items[3], 'type': 'str', 'value': self.find_function[self.items[3]], 'readonly': True},
            ]},
            {'name': 'Limitation', 'type': 'group', 'children': [
                {'name': self.items[4], 'type': 'int', 'value': self.find_function[self.items[4]], 'siPrefix': True, 'suffix': 'rpm', 'readonly': True},
                {'name': self.items[5], 'type': 'int', 'value': self.find_function[self.items[5]], 'readonly': True},
            ]},
            {'name': 'Read Sensor', 'type': 'group', 'children': [
                {'name': 'Motion Related', 'type': 'group', 'children': [
                    {'name': self.items[6], 'type': 'int', 'value': self.find_function[self.items[6]], 'siPrefix': True, 'suffix': 'rpm', 'readonly': True},
                    {'name': self.items[7], 'type': 'int', 'value': self.find_function[self.items[7]], 'readonly': True},
                    {'name': self.items[8], 'type': 'int', 'value': self.find_function[self.items[8]], 'readonly': True},
                ]},
                {'name': 'Error', 'type': 'group', 'children': [
                    {'name': self.items[9], 'type': 'str', 'value': self.find_function[self.items[9]], 'readonly': True},
                    {'name': self.items[10], 'type': 'str', 'value': self.find_function[self.items[10]], 'readonly': True},
                ]},
                {'name': 'Car Status', 'type': 'group', 'children': [
                    {'name': self.items[11], 'type': 'int', 'value': self.find_function[self.items[11]], 'readonly': True},
                    {'name': self.items[12], 'type': 'int', 'value': self.find_function[self.items[12]], 'readonly': True},
                    {'name': self.items[13], 'type': 'int', 'value': self.find_function[self.items[13]], 'readonly': True},
                    {'name': self.items[14], 'type': 'str', 'value': self.find_function[self.items[14]], 'readonly': True},
                    {'name': self.items[15], 'type': 'str', 'value': self.find_function[self.items[15]], 'readonly': True},
                ]},
            ]},
        ]

    def timer_event(self):
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(16)
        self.start()

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    ctrl = Controller()
    vis = BoxVisualizer(ctrl)
    vis.timer_event()
