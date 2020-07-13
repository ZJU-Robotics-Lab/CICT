# RoBoCar
This repository is for the robot car development, including main program, sensors test scripts, auto install scripts, helpful tools, etc.

![robocar](./doc/robocar.jpg)


#### Sensors used:
* LiDAR: [Velodyne VLP-16](https://www.velodynelidar.com/vlp-16.html)
* Camera: [MYNT EYE D-1000-120](https://www.myntai.com/mynteye/depth)
* GPS: [千寻 D300-GNSS](https://mall.qxwz.com/market/products/details?name=ouabiwv7762375598)
* IMU： [Xsens MTi-300-2A8G4](https://www.mouser.com/ProductDetail/Xsens/MTI-300-2A8G4?qs=sGAEpiMZZMutXGli8Ay4kNSxHzx9HmD09sFWWfMc%252BdM%3D)

# Install
```bash
cd scripts
bash install.sh
```

# Features
* Run deep learning model
```bash
python all_in_one.py
```

* Camera image reading in python with pybind11
```bash
bash camera/build.sh
python camera/run.py
```
* Camera calibration
```bash
cd scripts
python calibration.py --dir imgs --test left-0001.png
```
* LiDAR data reading
```bash
cd LiDAR/velodyne-driver
mkdir build
cd build
cmake ..
make -j16
sudo make install
cd ../../
python lidar.py
```
* IMU data reading
```bash
bash IMU/get_permission.sh
python IMU/mtnode.py
```
* XBox control
  * Right axis up: speed
  * Top axis left and right: rotation
  * Buttom Y: forward
  * Buttom A: backward
  * Buttom LOG: break and stop
  * Hat up and down: increase or reduce max speed
  * Hat right and left: increase or reduce acceleration
```bash
cd controller
bash get_permission.sh
python xbox.py
```



# ROS Wrapper
#### Build
```bash
cd ROS
bash build.sh
```

#### Launch
```bash
source devel/setup.bash
roslaunch mynteye_wrapper_d mynteye.launch
roslaunch velodyne_pointcloud VLP16_points.launch
python2 src/gps/scripts/run_gps.py
python2 src/controller/scripts/run.py
```

#### Record
```bash
rosbag record /mynteye/left/image_color /velodyne_points /gps /cmd
```

#### Read Rosbag
```bash
python2 ROS/src/tools/save_img.py -d 1 -b 2020-07-11-17-50-49.bag
```

#### Calibration
```bash
cd ROS
bash run_collect.sh
```
