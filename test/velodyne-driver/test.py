#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import numpy as np
import velodyne

data = b'6'*1200

while True:
    result = velodyne.parse_data(data)
    print(result[:,:3].shape)
    #time.sleep(1/(36000/2))
    break