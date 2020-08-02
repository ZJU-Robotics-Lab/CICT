
import numpy as np
import copy
import time

import carla
from .carla_sensor import CarlaSensorDataConversion, get_specific_sensor
from .camera.parameters import CameraParams, IntrinsicParams, ExtrinsicParams
from .camera.coordinate_transformation import CoordinateTransformation, rotationMatrix3D, reverseX, reverseY
from .camera import basic_tools


# from sensor_msgs.msg import PointCloud
# from geometry_msgs.msg import Point32
# import rospy

import sys
from ff.system import printVariable
from ff.system import env_path
# env_path.remove_python2_path(sys)
import cv2
# env_path.append_python2_path(sys)


class InversePerspectiveMapping(object):
    def __init__(self, param, sensor):
        # vehicle_width = param.vehicle_width
        # self.longitudinal_sample_number_near = param.longitudinal_sample_number_near
        # self.longitudinal_sample_number_far = param.longitudinal_sample_number_far
        

        # self.vehicle_half_width = vehicle_width / 2

        # for drawLineInImage
        # self.lateral_step_factor = param.lateral_step_factor
        # for drawLineInWorld
        # lateral_sample_number = param.lateral_sample_number
        # self.lateral_sample_array = np.linspace(-self.vehicle_half_width, self.vehicle_half_width, lateral_sample_number)
        
        self.sensor = sensor
        intrinsic_params = IntrinsicParams(sensor)
        extrinsic_params = ExtrinsicParams(sensor)
        self.camera_params = CameraParams(intrinsic_params, extrinsic_params)

        self.img_width = eval(sensor.attributes['image_size_x'])
        self.img_height = eval(sensor.attributes['image_size_y'])
        self.max_pixel = np.array([self.img_height, self.img_width]).reshape(2,1)
        self.min_pixel = np.array([0, 0]).reshape(2,1)

        self.empty_image = np.zeros((self.img_height, self.img_width, 3), dtype=np.dtype("uint8"))

        self.longitudinal_length = param.longitudinal_length
        self.ksize = param.ksize

        f = float(self.img_height) / self.longitudinal_length
        self.pesudo_K = np.array([  [f, 0, self.img_width/2],
                                    [0, f,  self.img_height],
                                    [0, 0,                1] ])
        self.reverseXY = basic_tools.np_dot(rotationMatrix3D(0,0,-np.pi/2))

        self.points_pub = rospy.Publisher('~points', PointCloud, queue_size=1)

        


    def getIPM(self, image):

        empty_image = copy.deepcopy(self.empty_image)

        index_array = np.argwhere(image == 255)
        index_array = index_array[:,:2]
        index_array = np.unique(index_array, axis=0)

        # point_cloud = PointCloud()
        # point_cloud.header.stamp = rospy.Time.now()
        # point_cloud.header.frame_id = 'base_link'

        tick_convert = 0.0

        for index in index_array:
            y_pixel, x_pixel = index[0], index[1]

            tick1 = time.time()

            image_vec = np.array([float(x_pixel), float(y_pixel)]).reshape(2,1)

            vehicle_vec = CoordinateTransformation.image2DToWorld3D(image_vec, self.camera_params.K, self.camera_params.R, self.camera_params.t)

            # homogeneous
            vehicle_vec[2,0] = 1.0
            new_image_vec = basic_tools.np_dot(self.pesudo_K, self.reverseXY, vehicle_vec)
            new_image_vec = new_image_vec[:2,:]
            new_image_vec = new_image_vec[::-1,:]

            if (new_image_vec >= self.min_pixel).all() and (new_image_vec < self.max_pixel).all():
                new_image_y_pixel, new_image_x_pixel = round(new_image_vec[0,0]), round(new_image_vec[1,0])
                empty_image[new_image_y_pixel, new_image_x_pixel, :] = 255

            tick2 = time.time()
            tick_convert += tick2 - tick1

            # point = Point32()
            # x, y = vehicle_vec[0,0], vehicle_vec[1,0]
            # point.x = x
            # point.y = y
            # point.z = 0
            # point_cloud.points.append(point)


        tick_1 = time.time()
        empty_image = cv2.GaussianBlur(empty_image, (self.ksize, self.ksize), 0)
        tick_2 = time.time()
        printVariable('time convert', tick_convert)
        printVariable('time gauss', tick_2 - tick_1)


        # self.points_pub.publish(point_cloud)

        return empty_image


