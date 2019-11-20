import cv2
from robot_camera import Camera

camera = Camera()

while True:
    img = camera.getImage()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR);
    cv2.imshow('input_image', img)
    cv2.waitKey(30)
    
cv2.destroyAllWindows()
# that will cause an error on Linux kernel version >= 4.16 in libeSPDI.so
# issue: https://github.com/slightech/MYNT-EYE-D-SDK/issues/13
camera.close()