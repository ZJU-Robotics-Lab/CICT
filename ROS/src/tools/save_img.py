import rosbag
import rospy
import cv2
from cv_bridge import CvBridge
from cv_bridge import CvBridgeError

img_path = '/home/wang/DataSet/yqdata/images2/'
gps_path = '/home/wang/DataSet/yqdata/gps2/'


class ImageCreator():
    def __init__(self):
        self.bridge = CvBridge()
        self.last_gps = ""
        
        self.gps_file = open(gps_path+'gps.txt', 'w')
        with rosbag.Bag('/home/wang/github/RoBoCar/ROS/2020-06-09-15-39-20.bag', 'r') as bag:
            for topic,msg,t in bag.read_messages():
                if topic == "/mynteye/left/image_color":
                    try:
                        cv_image = self.bridge.imgmsg_to_cv2(msg,"bgr8")
                    except CvBridgeError as e:
                        print(e)
                    timestr = "%.6f" %  msg.header.stamp.to_sec()
                    image_name = timestr+ ".png"
                    cv2.imwrite(img_path + image_name, cv_image)
                    
                    self.gps_file.write(timestr + '\t' + self.last_gps + '\n')
                elif topic == "/gps":
                    message_string = str(msg)
                    self.last_gps = message_string

if __name__ == '__main__':
    try:
        image_creator = ImageCreator()
    except rospy.ROSInterruptException:
        pass