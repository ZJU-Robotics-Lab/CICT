#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class Singleton(object):
    _instance = None

    def __new__(cls, *args, **kw):
        if not cls._instance:
            cls._instance = super(Singleton, cls).__new__(cls)  
        return cls._instance 
    
def debug(info, info_type='debug'):
	if info_type == 'error':
		print('\033[1;31m ERROR:', info, '\033[0m')
	elif info_type == 'success':
		print('\033[1;32m SUCCESS:', info, '\033[0m')
	elif info_type == 'warning':
		print('\033[1;34m WARNING:', info, '\033[0m')
	elif info_type == 'debug':
		print('\033[1;35m DEBUG:', info, '\033[0m')
	else:
		print('\033[1;36m MESSAGE:', info, '\033[0m')