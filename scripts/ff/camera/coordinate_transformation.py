
import numpy as np

from . import basic_tools


def rotationMatrix3D(roll, pitch, yaw):
    # RPY <--> XYZ, roll first, picth then, yaw final
    si, sj, sk = np.sin(roll), np.sin(pitch), np.sin(yaw)
    ci, cj, ck = np.cos(roll), np.cos(pitch), np.cos(yaw)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    R = np.identity(3)
    R[0, 0] = cj * ck
    R[0, 1] = sj * sc - cs
    R[0, 2] = sj * cc + ss
    R[1, 0] = cj * sk
    R[1, 1] = sj * ss + cc
    R[1, 2] = sj * cs - sc
    R[2, 0] = -sj
    R[2, 1] = cj * si
    R[2, 2] = cj * ci
    return R


def rotationMatrixRoll(roll):
    R = np.identity(3)
    R[1,1] = np.cos(roll)
    R[2,2] = np.cos(roll)
    R[2,1] = np.sin(roll)
    R[1,2] =-np.sin(roll)
    return R
def rotarotationMatrixPitch(pitch):
    R = np.identity(3)
    R[0,0] = np.cos(pitch)
    R[2,2] = np.cos(pitch)
    R[2,0] =-np.sin(pitch)
    R[0,2] = np.sin(pitch)
    return R
def rotarotationMatrixYaw(yaw):
    R = np.identity(3)
    R[0,0] = np.cos(yaw)
    R[1,1] = np.cos(yaw)
    R[1,0] = np.sin(yaw)
    R[0,1] =-np.sin(yaw)
    return R

def rotationMatrix3DYPR(roll, pitch, yaw):
    return basic_tools.np_dot(
            rotationMatrixRoll(roll),
            rotarotationMatrixPitch(pitch),
            rotarotationMatrixYaw(yaw)
            )

def reverseX():
    I = np.identity(3)
    I[0,0] = -1
    return I
def reverseY():
    I = np.identity(3)
    I[1,1] = -1
    return I


def intrinsicMatrix(fx, fy, u0, v0):
    K = np.array([  [fx, 0, u0],
                    [0, fy, v0],
                    [0,  0,  1] ])
    return K


class CoordinateTransformation(object):
    # I = rotationMatrix3D(-np.pi/2, 0, -np.pi/2).T
    '''
    when world_vec transforms into camera_vec, x axis of camera coordinate is vetical to 
    image plane, which should be z axis for intrinsic matrix, thus need this I matrix
    '''

    # I = rotationMatrix3D(0, 0, 0)

    # I = basic_tools.np_dot(reverseX(), rotationMatrix3DYPR(np.pi/2, 0, -np.pi/2))
    # I = rotationMatrix3DYPR(np.pi/2, 0, -np.pi/2)
    I = basic_tools.np_dot(reverseX(), reverseY(), rotationMatrix3DYPR(np.pi/2, 0, -np.pi/2))
    # I = basic_tools.np_dot(reverseY(), rotationMatrix3DYPR(np.pi/2, 0, -np.pi/2))

    @staticmethod
    def world3DToCamera3D(world_vec, R, t):
        '''
        Transforms a point from 'world coordinates' to 'camera coordinates'
        Args:
            world_vec: column vector (3,1), (x_W, y_W, z_W) [m]
            R: rotation matrix (3,3), (camera -> world coordinates)
            t: translation vector (3,1), (camera in world coordinates)
        Returns:
            camera_vec: column vector (3,1), (x_C, y_C, z_C) [m]
        '''
        # print('R: ' + str(R))
        # print('world_vec-t: ' + str(world_vec-t))
        # print('camera_vec: ' + str(np.dot(R.T , world_vec-t)))
        # print('\n---\n')
        camera_vec = basic_tools.np_dot(R.T, world_vec-t)
        # camera_vec = basic_tools.np_dot(R.T, world_vec) - t     # I think this is wrong
        return camera_vec
    @staticmethod
    def camera3DToWorld3D(camera_vec, R, t):
        '''
        Transforms a point from 'camera coordinates' to 'world coordinates'
        Args:
            camera_vec: column vector (3,1), (x_C, y_C, z_C) [m]
            R: rotation matrix (3,3), (camera -> world coordinates)
            t: translation vector (3,1), (camera in world coordinates)
        Returns:
            world_vec: column vector (3,1), (x_W, y_W, z_W) [m]
        '''
        world_vec = basic_tools.np_dot(R, camera_vec) + t   # TODO
        return world_vec


    @staticmethod
    def camera3DToImage2D(camera_vec, K, eps=1e-24):
        '''
        Transforms a point from 'camera coordinates' to 'image coordinates'
        Args:
            camera_vec: column vector (3,1), (x_C, y_C, z_C) [m]
            K: intrinsic matrix (3,3), (camera -> image coordinates)
        Returns:
            image_vec: column vector (2,1), (x_I, y_I) [px]
        '''
        image_vec = basic_tools.np_dot(K, CoordinateTransformation.I, camera_vec)
        # print('image_vec: '+str(image_vec))
        return image_vec[:2,:] / (image_vec[2,:] + eps)


    @staticmethod
    def world3DToImage2D(world_vec, K, R, t):
        camera_vec = CoordinateTransformation.world3DToCamera3D(world_vec, R, t)
        # print('t: '+str(t))
        # print('world_vec: '+str(world_vec))
        # print('camera_vec: '+str(camera_vec))
        # print()
        image_vec = CoordinateTransformation.camera3DToImage2D(camera_vec, K)
        return image_vec
    @staticmethod
    def world3DToImagePixel2D(world_vec, K, R, t):
        image_vec = CoordinateTransformation.world3DToImage2D(world_vec, K, R, t)
        x_pixel, y_pixel = round(image_vec[0,0]), round(image_vec[1,0])
        return np.array([x_pixel, y_pixel]).reshape(2,1)


    @staticmethod
    def image2DToWorld3D(image_vec, K, R, t):
        r = np.vstack((image_vec, 1))
        b = np.vstack(( basic_tools.np_dot(K, CoordinateTransformation.I, t), 0 ))
        A = np.vstack((np.hstack((basic_tools.np_dot(K, CoordinateTransformation.I, R.T), -r)), np.array([0,0,1,0]).reshape(1,4)))
        world_vec = np.dot(np.linalg.inv(A), b)
        return world_vec[:3]