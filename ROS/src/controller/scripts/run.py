# -*- coding: utf-8 -*-
from __future__ import print_function
from can_bus import Controller
from xbox import scan_usb, JoyStick

import rospy
from std_msgs.msg import String, Header
from controller.msg import Cmd


if __name__ == '__main__':
    CAN_CHANNEL = scan_usb('CAN')
    print('CAN_CHANNEL', CAN_CHANNEL)
    ctrl = Controller(CAN_CHANNEL)
    ctrl.start()
    joystick = JoyStick(ctrl, verbose=False)
    joystick.start()
    pub = rospy.Publisher('cmd', Cmd, queue_size=1)
    rospy.init_node('talker', anonymous=True)
    r = rospy.Rate(60)
    frame_id = "cmd_data"
    while not rospy.is_shutdown():
        v, w = joystick.get_data()
        data  =str(v) + '\t' +str(w) + '\n'

        h = Header()
        h.stamp = rospy.Time.now()
        h.frame_id = frame_id
        cmd_msg = Cmd()
        cmd_msg.header = h
        cmd_msg.data = data
        if data == None:
            continue
        #print(data)
        pub.publish(cmd_msg)
        r.sleep()