import cv2
import numpy as np
from .camera.parameters import CameraParams, IntrinsicParams, ExtrinsicParams
from .camera.coordinate_transformation import CoordinateTransformation, rotationMatrix3D#, reverseX, reverseY
from .camera import basic_tools

class InversePerspectiveMapping(object):
    def __init__(self, param, sensor):
        self.sensor = sensor
        intrinsic_params = IntrinsicParams(sensor)
        extrinsic_params = ExtrinsicParams(sensor)
        self.camera_params = CameraParams(intrinsic_params, extrinsic_params)

        self.img_width = 400#eval(sensor.attributes['image_size_x'])
        self.img_height = 200#eval(sensor.attributes['image_size_y'])
        #self.max_pixel = np.array([self.img_height, self.img_width]).reshape(2,1)
        #self.min_pixel = np.array([0, 0]).reshape(2,1)

        self.empty_image = np.zeros((self.img_height, self.img_width), dtype=np.dtype("uint8"))

        self.longitudinal_length = param.longitudinal_length
        self.ksize = param.ksize

        f = float(self.img_height) / self.longitudinal_length
        self.pesudo_K = np.array([  [f, 0, self.img_width/2],
                                    [0, f,  self.img_height],
                                    [0, 0,                1] ])
        self.reverseXY = basic_tools.np_dot(rotationMatrix3D(0,0,-np.pi/2))

    
    def getIPM(self, image):
        self.empty_image = np.zeros((self.img_height, self.img_width), dtype=np.dtype("uint8"))
        index_array = np.argwhere(image > 200)
        index_array = index_array[:,:2]
        index_array = np.unique(index_array, axis=0)
        index_array = np.array([index_array[:,1], index_array[:,0]])
        vehicle_vec = CoordinateTransformation.image2DToWorld3D2(index_array, self.camera_params.K, self.camera_params.R, self.camera_params.t)
        
        vehicle_vec[:,2,0] = 1.0
        temp = np.dot(self.pesudo_K, self.reverseXY)
        vehicle_vec = np.squeeze(vehicle_vec, axis = 2)
        new_image_vec = np.dot(temp, vehicle_vec.T)
        new_image_vec = new_image_vec[:2,:]
        new_image_vec = new_image_vec[::-1,:]

        new_image_y_pixel = new_image_vec[0,:].astype(int)
        new_image_x_pixel = new_image_vec[1,:].astype(int)
        
        #self.empty_image[new_image_y_pixel, new_image_x_pixel] = 255

        mask = np.where((new_image_x_pixel >= 0)&(new_image_x_pixel < self.img_width))[0]
        new_image_x_pixel = new_image_x_pixel[mask]
        new_image_y_pixel = new_image_y_pixel[mask]
        
        mask = np.where((new_image_y_pixel >= 0)&(new_image_y_pixel < self.img_height))[0]
        new_image_x_pixel = new_image_x_pixel[mask]
        new_image_y_pixel = new_image_y_pixel[mask]
        self.empty_image[new_image_y_pixel, new_image_x_pixel] = 255

        self.empty_image[np.clip(new_image_y_pixel+1,0, self.img_height-1),new_image_x_pixel] = 255
        self.empty_image[np.clip(new_image_y_pixel-1,0, self.img_height-1),new_image_x_pixel] = 255
        
        #self.empty_image = cv2.GaussianBlur(self.empty_image, (self.ksize, self.ksize), 25)
        return self.empty_image