# RoBoCar
This repository is for the robot car development, including main program, sensors test scripts, auto install scripts, helpful tools, etc.
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
python LiDAR/lidar.py
```
* IMU data reading
```bash
bash IMU/get_permission.sh
python IMU/mtnode.py
```

# Test List
- [x] GPS data reading
- [x] CAN-USB test with python-can
