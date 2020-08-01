
# carla_sensor.py

# import rospy, tf
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from sensor_msgs.point_cloud2 import create_cloud_xyz32
from geometry_msgs.msg import TransformStamped

import sys
# sys.path.insert(0, '/home/zdk/miniconda3/envs/ros_demo/lib/python3.7/site-packages')
# sys.path.insert(0, '/home/zdk/miniconda3/envs/ros_demo/catkin_ws/devel_isolated/cv_bridge/lib/python3/dist-packages')

import cv2
from cv_bridge import CvBridge


import carla
from carla import ColorConverter as cc

import numpy as np
import weakref
import copy
import time

def get_sensors(world, vehicle):
    actor_list = world.get_actors()
    sensor_list = []
    for actor in actor_list:
        if actor.parent is not None and actor.parent.type_id == vehicle.type_id:
            sensor_list.append(actor)
    return sensor_list

def get_specific_sensor(world, vehicle, sensor_type):
    sensor_list = get_sensors(world, vehicle)
    for sensor in sensor_list:
        if sensor_type in sensor.type_id:
            return sensor
    return None





class CarlaSensorListMaster(object):
    def __init__(self):
        self.sensor_list = []

    def append(self, sensor, transform, binded):
        sensor_master = CarlaSensorMaster(sensor, transform, binded)
        self.sensor_list.append(sensor_master)

    def destroy(self):
        for sensor_master in self.sensor_list:
            sensor_master.destroy()


class Sensor(object):
    def __init__(self, transform, config):
        self.type_id = 'sensor.camera.rgb'
        self.transform = transform
        self.attributes = dict()
        self.attributes['role_name'] = 'front'
        self.attributes['image_size_x'] = str( config['img_length'] )
        self.attributes['image_size_y'] = str( config['img_width'] )
        self.attributes['fov'] = str( config['fov'] )


class CarlaSensorMaster(object):
    def __init__(self, sensor, transform, binded):
        self.sensor = sensor
        self.transform = transform
        self.raw_data, self.data = None, None

        self.type_id = sensor.type_id
        self.attributes = sensor.attributes

        weak_self = weakref.ref(self)
        if 'lidar' in sensor.type_id:
            self.frame_id = 'lidar/{}'.format(sensor.attributes['role_name'])
            if binded is False:
                self.callback = lambda data: CarlaSensorCallback.lidar(weak_self, data)
                self.sensor.listen(self.callback)
        elif 'camera' in sensor.type_id:
            self.frame_id = 'camera_rgb/{}'.format(sensor.attributes['role_name'])
            if binded is False:
                self.callback = lambda data: CarlaSensorCallback.camera_rgb(weak_self, data)
                self.sensor.listen(self.callback)
        elif 'gnss' in sensor.type_id:
            self.frame_id = 'gnss/{}'.format(sensor.attributes['role_name'])
            if binded is False:
                self.callback = lambda data: CarlaSensorCallback.gnss(weak_self, data)
                self.sensor.listen(self.callback)
        elif 'imu' in sensor.type_id:
            self.frame_id = 'imu/{}'.format(sensor.attributes['role_name'])
            if binded is False:
                self.callback = lambda data: CarlaSensorCallback.imu(weak_self, data)
                self.sensor.listen(self.callback)


    def get_transform(self):
        return self.transform


    def publish(self, data):
        if 'camera' in self.type_id and self.ros_pub is True:
            # data: the result of cv2.imread, np.array
            image = CarlaSensorDataConversion.cv2ImageToSensorImage(data, self.frame_id)
            self.publisher.publish(image)



    def get_tf_stamped(self, vehicle_frame_id):
        tf_stamped = TransformStamped()
        tf_stamped.header.stamp = rospy.Time.now()
        tf_stamped.header.frame_id = vehicle_frame_id
        tf_stamped.child_frame_id = self.frame_id
        tf_stamped.transform.translation.x = self.transform.location.x
        tf_stamped.transform.translation.y = self.transform.location.y
        tf_stamped.transform.translation.z = self.transform.location.z
        roll, pitch, yaw = np.deg2rad(self.transform.rotation.roll), np.deg2rad(self.transform.rotation.pitch), np.deg2rad(self.transform.rotation.yaw)
        quaternion = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
        tf_stamped.transform.rotation.x = quaternion[0]
        tf_stamped.transform.rotation.y = quaternion[1]
        tf_stamped.transform.rotation.z = quaternion[2]
        tf_stamped.transform.rotation.w = quaternion[3]
        return tf_stamped



    def destroy(self):
        self.sensor.destroy()





class CarlaSensorCallback(object):
    @staticmethod
    def lidar(weak_self, data):
        # data: carla.LidarMeasurement
        self = weak_self()
        self.raw_data = data
        self.data = CarlaSensorDataConversion.lidarMeasurementToPointcloud2(data, self.frame_id)
        print('callback: ' + str(type(self.data)))

    @staticmethod
    def camera_rgb(weak_self, data):
        # data: carla.Image
        self = weak_self()
        self.raw_data = data
        self.data = CarlaSensorDataConversion.carlaImageToSensorImage(data, self.frame_id)

    @staticmethod
    def gnss(weak_self, data):
        # data: carla.GNSSMeasurement
        self = weak_self()
        self.raw_data = data
        self.data = data

    @staticmethod
    def imu(weak_self, data):
        # data: carla.IMUMeasurement
        self = weak_self()
        self.raw_data = data
        self.data = data


class CarlaSensorDataConversion(object):
    cv_bridge = CvBridge()
    @staticmethod
    def lidarMeasurementToPointcloud2(lidar_measurement, frame_id):
        lidar_data = np.frombuffer(lidar_measurement.raw_data, dtype=np.float32)
        lidar_data = np.reshape(lidar_data, (int(lidar_data.shape[0] / 3), 3))
        # we take the oposite of y axis
        # (as lidar_measurement point are express in left handed coordinate system, and ros need right handed)
        # we need a copy here, because the data are read only in carla numpy
        # array

        lidar_data = -lidar_data
        lidar_data = copy.copy(lidar_data)
        # lidar_data.flags.writeable = True
        lidar_data[:,[0]] *= -1
        # we also need to permute x and y
        lidar_data = lidar_data[..., [1, 0, 2]]
        
        header = Header()
        header.frame_id = frame_id
        header.stamp = rospy.Time.now()
        point_cloud_msg = create_cloud_xyz32(header, lidar_data)
        return point_cloud_msg

    @staticmethod
    def lidarMeasurementToNPArray(lidar_measurement):
        lidar_data = np.frombuffer(lidar_measurement.raw_data, dtype=np.float32).reshape([-1, 3])
        point_cloud = np.stack([-lidar_data[:,1], -lidar_data[:,0], -lidar_data[:,2]])
        mask = np.where((point_cloud[0] > 1.0)|(point_cloud[0] < -4.0)|(point_cloud[1] > 1.2)|(point_cloud[1] < -1.2))[0]
        point_cloud = point_cloud[:, mask]
        return point_cloud

    # TODO
    @staticmethod
    def NPArrayToPointcloud2(np_array, frame_id):
        header = Header()
        header.frame_id = frame_id
        # print('frame: '+str(frame_id)+',  '+str(np_array.shape))
        header.stamp = rospy.Time.now()
        t1 = time.time()
        np_array_T = np_array.T
        t2 = time.time()
        point_cloud_msg = create_cloud_xyz32(header, np_array_T)
        t3 = time.time()
        print('    time 1: ' + str(t2-t1))
        print('    time 2: ' + str(t3-t2))
        return point_cloud_msg



    @staticmethod
    def carlaImageToSensorImage(carla_image, frame_id):
        carla_image.convert(cc.Raw)
        # array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        # array = np.reshape(array, (image.height, image.width, 4))
        # array = array[:, :, :3]
        # array = array[:, :, ::-1]
        # self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        # ff: for visualization
        # self.image = image

        # from sensor_msgs.msg import Image
        # image = Image()

        # rospy.loginfo('carla_image'+str(carla_image))

        # print('here')
        image_data_array, encoding = CarlaSensorDataConversion.get_carla_image_data_array(carla_image=carla_image)
        image = CarlaSensorDataConversion.cv_bridge.cv2_to_imgmsg(image_data_array, encoding=encoding)
        # the camera data is in respect to the camera's own frame
        image.header.stamp = rospy.Time.now()
        image.header.frame_id = frame_id
        return image

    @staticmethod
    def get_carla_image_data_array(carla_image):
        """
        Function (override) to convert the carla image to a numpy data array
        as input for the cv_bridge.cv2_to_imgmsg() function

        The RGB camera provides a 4-channel int8 color format (bgra).

        :param carla_image: carla image object
        :type carla_image: carla.Image
        :return tuple (numpy data array containing the image information, encoding)
        :rtype tuple(numpy.ndarray, string)
        """

        carla_image_data_array = np.ndarray(
            shape=(carla_image.height, carla_image.width, 4),
            dtype=np.uint8, buffer=carla_image.raw_data)

        return carla_image_data_array, 'bgra8'


    # TODO
    @staticmethod
    def NPArrayToSensorImage(np_array, frame_id):
        # np_array = np_array.swapaxes(0, 1)
        image = CarlaSensorDataConversion.cv_bridge.cv2_to_imgmsg(np_array, encoding='bgr8')
        # the camera data is in respect to the camera's own frame
        image.header.stamp = rospy.Time.now()
        image.header.frame_id = frame_id
        return image

    @staticmethod
    def cv2ImageToSensorImage(np_array, frame_id):
        # np_array = np_array.swapaxes(0, 1)
        image = CarlaSensorDataConversion.cv_bridge.cv2_to_imgmsg(np_array, encoding='bgr8')
        # the camera data is in respect to the camera's own frame
        image.header.stamp = rospy.Time.now()
        image.header.frame_id = frame_id
        return image