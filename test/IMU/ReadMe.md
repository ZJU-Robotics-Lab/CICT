From: https://github.com/rpng/xsens_standalone

# xsens_standalone

This is just a small program that does not depend on the ROS framework that allows for communication with the xsens imus. Most code was taken from [ethz-asl/ethzasl_xsens_driver](https://github.com/ethz-asl/ethzasl_xsens_driver) package, but all version of ROS have been removed. In the main script there is a way to specify what measurements should be reported from the IMU. Currently the linear acceleration, angular velocity, magnetometer, and orientation are being recorded.


## File overview

* mtdef.py - Contains all protocol variables needed
* mtdevice.py - Actual python driver, opens a serial port with the IMU
* mtnode.py - Wrapper that will try to get a new message from IMU as soon as possible


## Configuration

One can configure the IMU using a series of strings. This information can also be found in the bottom of the `mtdevice.py` file. One can then call `mtdevice.get_output_config(args)` to get the config file, and then finally configure the IMU by calling `SetOutputConfiguration()`. 


```
The format is a sequence of "<group><type><frequency>?<format>?"
separated by commas.
The frequency and format are optional.
The groups and types can be:
    t  temperature (max frequency: 1 Hz):
        tt  temperature
    i  timestamp (max frequency: 2000 Hz):
        iu  UTC time
        ip  packet counter
        ii  Integer Time of the Week (ITOW)
        if  sample time fine
        ic  sample time coarse
        ir  frame range
    o  orientation data (max frequency: 400 Hz):
        oq  quaternion
        om  rotation matrix
        oe  Euler angles
    b  pressure (max frequency: 50 Hz):
        bp  baro pressure
    a  acceleration (max frequency: 2000 Hz (see documentation)):
        ad  delta v
        aa  acceleration
        af  free acceleration
        ah  acceleration HR (max frequency 1000 Hz)
    p  position (max frequency: 400 Hz):
        pa  altitude ellipsoid
        pp  position ECEF
        pl  latitude longitude
    n  GNSS (max frequency: 4 Hz):
        np  GNSS PVT data
        ns  GNSS satellites info
    w  angular velocity (max frequency: 2000 Hz (see documentation)):
        wr  rate of turn
        wd  delta q
        wh  rate of turn HR (max frequency 1000 Hz)
    g  GPS (max frequency: 4 Hz):
        gd  DOP
        gs  SOL
        gu  time UTC
        gi  SV info
    r  Sensor Component Readout (max frequency: 2000 Hz):
        rr  ACC, GYR, MAG, temperature
        rt  Gyro temperatures
    m  Magnetic (max frequency: 100 Hz):
        mf  magnetic Field
    v  Velocity (max frequency: 400 Hz):
        vv  velocity XYZ
    s  Status (max frequency: 2000 Hz):
        sb  status byte
        sw  status word
Frequency is specified in decimal and is assumed to be the maximum
frequency if it is omitted.
Format is a combination of the precision for real valued numbers and
coordinate system:
    precision:
        f  single precision floating point number (32-bit) (default)
        d  double precision floating point number (64-bit)
    coordinate system:
        e  East-North-Up (default)
        n  North-East-Down
        w  North-West-Up
Examples:
    The default configuration for the MTi-1/10/100 IMUs can be
    specified either as:
        "wd,ad,mf,ip,if,sw"
    or
        "wd2000fe,ad2000fe,mf100fe,ip2000,if2000,sw2000"
    For getting quaternion orientation in float with sample time:
        "oq400fw,if2000"
    For longitude, latitude, altitude and orientation (on MTi-G-700):
        "pl400fe,pa400fe,oq400fe"
```
