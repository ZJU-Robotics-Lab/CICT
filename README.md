# RoBoCar
This repository is for the robot car development, including main program, sensors test scripts, auto install scripts, helpful tools, etc.
#### Sensors used:
* LiDAR: [Velodyne VLP-16](https://www.velodynelidar.com/vlp-16.html)
* Camera: [MYNT EYE D-1000-120](https://www.myntai.com/mynteye/depth)
* GPS: [千寻 D300-GNSS](https://mall.qxwz.com/market/products/details?name=ouabiwv7762375598)
* IMU： [Xsens MTi-3](https://shop.xsens.com/shop/mti-1-series/mti-3-ahrs/mti-3)


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


# Test List
- [x] LiDAR data reading in python
- [ ] GPS data reading
- [x] CAN-USB test with python-can
- [x] auto install scripts on ubuntu16.04/18.04

# TODO List
- [ ] IMU data reading
