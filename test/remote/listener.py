#!/usr/bin/env python
import time
import rospy
import base64
import numpy as np
from std_msgs.msg import String
from informer import Informer, config

config.PUBLICT_IP = '127.0.0.1'
config.PORT_DICT = {'message':10006,}
config.RECV_KEYS = ['message']
config.REGISTER_KEYS = list(config.PORT_DICT.keys())

class Talker(Informer):
    def parse_message(self, message):
        message_type = message['Mtype']
        pri = message['Pri']
        robot_id = message['Id']
        data = message['Data']
        print(message_type, pri, robot_id, data)
        
ifm = Talker(robot_id="listen", block=True)

def callback(data):
    data = np.random.rand(1000, 4)
    encode_data = base64.b64encode(data.tostring()).decode()
    ifm.send_message(encode_data)
    print('Get ros message and send to the cloud', time.time())
     
def listener():
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("chatter", String, callback)
    rospy.spin()
 
if __name__ == '__main__':
    listener()