import os
import cv2
import glob
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='.', help="image files dir")
parser.add_argument('--test', type=str, default=None, help='test image name')
opt = parser.parse_args()

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob(opt.dir+'/*.png')
images2 = glob.glob(opt.dir+'/*.jpg')
images.extend(images2)
assert len(images) > 0
print('Find images', len(images))

gray = None
for fname in images:
    img = cv2.imread(fname)
    assert type(img) != type(None)

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7,6),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)
        cv2.imshow('img', img)
        cv2.waitKey(50)

cv2.destroyAllWindows()
# return the camera matrix, distortion coefficients, rotation and translation vectors
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
print('camera matrix:', mtx, '\ndistortion coefficients', dist[0])

######################### error ################################
mean_error = 0
tot_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    tot_error += error

assert len(objpoints) > 0
print("total error:", mean_error/len(objpoints))

######################### test ################################
if opt.test != None:
    file_path = opt.dir + '/' + opt.test
    if os.path.exists(file_path):
        img = cv2.imread(file_path)
        assert type(img) != type(None)

        h,  w = img.shape[:2]
        newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
        
        # undistort
        mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
        dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
        
        # crop the image
        x,y,w,h = roi
        dst = dst[y:y+h, x:x+w]
        #cv2.imwrite('calibresult.png',dst)
    
        cv2.imshow('origin',img)
        cv2.imshow('calibration',dst)
        cv2.waitKey(5000)
        cv2.destroyAllWindows()
    else:
        print(file_path, 'not exist !')