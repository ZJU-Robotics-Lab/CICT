
import os
import sys
import glob

def load(path):
    try:
        sys.path.append(path+'/PythonAPI')
        sys.path.append(glob.glob(path+'/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
            sys.version_info.major,
            sys.version_info.minor,
            'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    except:
        print('Fail to load carla library')