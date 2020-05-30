import cv2
import numpy as np

path = "/home/wang/Downloads/collect_data/"
index = '14'

def read_pcd(file_path):
    x = []
    y = []
    z = []
    intensity = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        [lines.pop(0) for _ in range(9)]
        for line in lines:
            sp_line = line.split()
            x.append(float(sp_line[0]))
            y.append(float(sp_line[1]))
            z.append(float(sp_line[2]))
            intensity.append(float(sp_line[3]))
    return np.array([x, y, z]), intensity

width = 1280
height = 720

fx = 711.642238
fy = 711.302135
s = 0.0
x0 = 644.942373
y0 = 336.030580

cameraMat = np.array([
        [fx,  s, x0],
        [0., fy, y0],
        [0., 0., 1.]
])

distortionMat = np.array([-0.347125, 0.156284, 0.001037, -0.000109 ,0.000000])

img = cv2.imread(path+'left_'+index+'.png')
#newcameramtx, roi = cv2.getOptimalNewCameraMatrix(cameraMat, distortionMat, (width, height), 1, (width, height))
#undistort_img = cv2.undistort(img, cameraMat, distortionMat, None, newcameramtx)
#cv2.imwrite('calibresult.png',undistort_img) 

RTMat = np.array([
    [-0.0024, -1.0000, -0.0033, 0.0660],
    [0.0746,  0.0031,  -0.9972, 0.1263],
    [0.9972,  -0.0026, 0.0746,  0.2481],
    [0.0,     0.0,     0.0,     1.0]
])

point_cloud, intensity = read_pcd(path+'cloud_'+index+'.pcd')


rotationMat = np.array([
    [-0.0024, -1.0000, -0.0033],
    [0.0746,  0.0031,  -0.9972],
    [0.9972,  -0.0026, 0.0746],
])

traslationMat = np.array([0.0660, 0.1263, 0.2481])

trans_pc = np.dot(rotationMat, point_cloud) + np.tile(traslationMat, (point_cloud.shape[1], 1)).T

image_uv = np.array([
        trans_pc[0]*fx/trans_pc[2] + x0,
        trans_pc[1]*fy/trans_pc[2] + y0
        ])


point_size = 1
point_color = (0, 0, 255) # BGR
thickness = 4
for i in range(image_uv.shape[1]):
    point = (int(image_uv[0][i]), int(image_uv[1][i]))
    if point[0] > width or point[0] < 0 or point[1] > height or point[1] < 0:
        continue
    if intensity[i] < 10: continue
    cv2.circle(img, point, point_size, point_color, thickness)
    
cv2.imwrite('merge_'+index+'.png',img) 

"""

rotation = cameraMat*rotationMat
translation = cameraMat*traslationMat
R = rotation
T = translation
roadheight = -1.3

path_xy = np.array([
        [1.3, 1.3],
        [0.5, 0.2],
        [0.7, 0.25],
        ])

path_z = np.array([[roadheight]]*path_xy.shape[0])

path_xyz = np.concatenate((path_xy, path_z), axis = 1)

uvz = R*path_xyz + np.tile(traslationMat,(path_xy.shape[0], 1))

path_uv = uvz[:,:2] / np.tile(uvz[:,2:], [1, 2])
print(path_uv)

"""