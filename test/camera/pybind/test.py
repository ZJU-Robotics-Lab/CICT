import robot_camera
import cv2
import matplotlib.pyplot as plt

#image_rgb = cv2.imread('lena_rgb.jpg', cv2.IMREAD_UNCHANGED)

#var1 = robot_camera.test_rgb_to_gray(image_rgb)
img = robot_camera.test_read_img()
print(img.shape)
plt.figure('rgb-gray')
plt.imshow(img)