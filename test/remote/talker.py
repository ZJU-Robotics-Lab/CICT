#!/usr/bin/env python
import time
import base64
import numpy as np
import rospy
from std_msgs.msg import String
from informer import Informer, config

config.PUBLICT_IP = '127.0.0.1'
config.PORT_DICT = {'message':10006,}
config.RECV_KEYS = ['message']
config.REGISTER_KEYS = list(config.PORT_DICT.keys())

class Talker(Informer):
    def __init__(self, robot_id=None, block=True):
        self.pub = rospy.Publisher('chatter', String, queue_size=1)
        rospy.init_node('talker', anonymous=True)
        super().__init__(robot_id, block)

        
    def parse_message(self, message):
        #message_type = message['Mtype']
        #pri = message['Pri']
        #robot_id = message['Id']
        data = message['Data']
        decod_data = base64.b64decode(data)
        array = np.frombuffer(decod_data,dtype=float).reshape(1000, 4)
        print('Parse result:\n', array)
        self.pub.publish(data)
        print('Get message from the cloud and send to ROS', time.time())


if __name__ == '__main__':
    try:
        ifm = Talker(robot_id="talker", block=True)
        ifm.pub.publish('START')
        while True:
            time.sleep(1)
    except rospy.ROSInterruptException:
        pass