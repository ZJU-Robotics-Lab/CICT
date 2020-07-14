import cv2
import time
import numpy as np
from get_nav import NavMaker
from sensor_manager import SensorManager, scan_usb

sensor_dict = {
        'gps':None,
        'imu':None,
        }

sm = SensorManager(sensor_dict)
sm.init_all()
nav_maker = NavMaker(sm['gps'], sm['imu'])
nav_maker.start()

def get_nav():
    global nav_maker
    nav = nav_maker.get()
    return nav

"""
for i in range(100):
	print(i)
	time.sleep(0.01)
	nav = get_nav()
nav.show()
"""
time.sleep(1)
while True:
	nav = get_nav()
	img = np.array(nav)
	img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
	cv2.imshow('Nav', img)
	if cv2.waitKey(10) & 0xFF == ord('q'):
		break

cv2.destroyAllWindows()