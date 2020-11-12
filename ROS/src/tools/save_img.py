import rosbag
import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
import argparse

parser = argparse.ArgumentParser(description='Params')
parser.add_argument('-d', '--data', type=int, default=3, help='data index')
parser.add_argument('-b', '--bag', type=str, default="2020-10-10-16-58-22.bag", help='bag name')
args = parser.parse_args()

img_path = '/media/wang/Data1/images'+str(args.data)+'/'
gps_path = '/media/wang/Data1/gps'+str(args.data)+'/'
cmd_path = '/media/wang/Data1/cmd'+str(args.data)+'/'

class ImageCreator():
    def __init__(self):
        self.bridge = CvBridge()
        self.last_gps = ""
        self.last_cmd = ""
        
        self.gps_file = open(gps_path+'gps.txt', 'w')
        self.cmd_file = open(cmd_path+'cmd.txt', 'w')
        with rosbag.Bag('/media/wang/Data1/'+args.bag, 'r') as bag:
            for topic,msg,t in bag.read_messages():
                if topic == "/mynteye/left/image_color":
                    try:
                        cv_image = self.bridge.imgmsg_to_cv2(msg,"bgr8")
                    except CvBridgeError as e:
                        print(e)
                    timestr = "%.6f" %  msg.header.stamp.to_sec()
                    image_name = timestr+ ".png"
                    cv2.imwrite(img_path + image_name, cv_image)
                    
                    self.gps_file.write(timestr + '\t' + self.last_gps)
                    self.cmd_file.write(timestr + '\t' + self.last_cmd)
                elif topic == "/gps":
                    message_string = str(msg.data)
                    self.last_gps = message_string
                elif topic == "/cmd":
                    message_string = str(msg.data)
                    self.last_cmd = message_string

if __name__ == '__main__':
    try:
        image_creator = ImageCreator()
    except rospy.ROSInterruptException:
        pass