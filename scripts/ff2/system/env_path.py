
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'

def remove_ros_path(sys):
    if ros_path in sys.path:
        sys.path.remove(ros_path)

def append_ros_path(sys):
    if ros_path not in sys.path:
        sys.path.append(ros_path)

python2_path = []
def remove_python2_path(sys):
    for i, p in enumerate(sys.path):
        if 'python2' in p:
            python2_path.append(p)
            del sys.path[i]

def append_python2_path(sys):
    sys.path.extend(python2_path)


def append(sys, path):
	sys.path.append(path)