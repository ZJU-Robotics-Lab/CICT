
import numpy as np
import copy

from .carla_sensor import CarlaSensorDataConversion, get_specific_sensor
from .camera.parameters import CameraParams, IntrinsicParams, ExtrinsicParams
from .camera.coordinate_transformation import CoordinateTransformation

class CollectPerspectiveImage(object):
    def __init__(self, traj_length, target_speed, fps, world, vehicle):
        
        # print('[CollectPerspectiveImage] sensor attributes: '+str(sensor.attributes))
        self.vehicle = vehicle
        self.vehicle_half_width = vehicle.bounding_box.extent.y
        sample_number = round(self.vehicle_half_width / 0.01)+1   # sample along wdith
        self.sample_array = np.linspace(-self.vehicle_half_width, self.vehicle_half_width, sample_number)

        self.sensor = get_specific_sensor(world, vehicle, 'camera')
        self.intrinsic_params = IntrinsicParams(self.sensor)

        img_width = eval(self.sensor.attributes['image_size_x'])
        img_height = eval(self.sensor.attributes['image_size_y'])
        # print(type(img_height))
        self.max_pixel = np.array([img_height, img_width]).reshape(2,1)
        self.min_pixel = np.array([0, 0]).reshape(2,1)

        self.size = round((float(traj_length) / target_speed) * fps) + 1
        print('[CollectPerspectiveImage] size: '+str(self.size))
        self.time_stamp_list = []
        self.camera_params_list = []
        self.pose_list = []
        self.image_list = []


    def getPM(self, time_stamp, pose, image):
        '''
        Args: 
            time_stamp: time.time()
            pose: carla.Transform
        '''
        self.time_stamp_list.append(copy.deepcopy(time_stamp))

        camera_params = CameraParams(self.intrinsic_params, ExtrinsicParams(self.sensor))
        self.camera_params_list.append(camera_params)

        self.pose_list.append(pose)
        self.image_list.append(copy.deepcopy(image))

        print('size: '+str(len(self.image_list)))
        if len(self.image_list) <= self.size:
            return None

        del self.time_stamp_list[0]
        del self.camera_params_list[0]
        del self.pose_list[0]
        del self.image_list[0]


        empty_image = copy.deepcopy(self.image_list[0])

        # clear
        # empty_image[:,:,:] = 0
        # print("vehicle: " + str(pose))
        # print("camera: " + str(self.sensor.get_transform()))


        param = self.camera_params_list[0]
        for pose in self.pose_list:
            vec = np.array([pose.location.x, pose.location.y, pose.location.z]).reshape(3,1)
            theta = np.deg2rad(pose.rotation.yaw - 90)

            for distance in self.sample_array:
                world_vec = np.array([distance*np.cos(theta), distance*np.sin(theta), 0]).reshape(3,1) + vec

                # print("vehicle: " + str(pose))
                # print("camera: " + str(self.sensor.get_transform()))

                # world_vec = np.array([pose.location.x, pose.location.y, pose.location.z]).reshape(3,1)
                image_pixel_vec = CoordinateTransformation.world3DToImagePixel2D(world_vec, param.K, param.R, param.t)
                image_pixel_vec = image_pixel_vec[::-1,:]
                if (image_pixel_vec >= self.min_pixel).all() and (image_pixel_vec < self.max_pixel).all():
                    # print('here')
                    # print('max_pixel: ' + str(self.max_pixel))
                    # print('image_pixel_vec: ' + str(image_pixel_vec))
                    # print('shape: '+str(empty_image.shape))
                    # print()

                    x_pixel, y_pixel = image_pixel_vec[0,0], image_pixel_vec[1,0]
                    empty_image[x_pixel,y_pixel,:] = 0

        return empty_image












