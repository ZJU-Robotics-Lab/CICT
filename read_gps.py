# -*- coding: utf-8 -*-
from __future__ import print_function
from gps import GPS

gps = GPS()
gps.start()

while True:
	latitude,longtitude = gps.get()
	print(latitude,longtitude)