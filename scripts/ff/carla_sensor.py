import weakref

def get_sensors(world, vehicle):
    actor_list = world.get_actors()
    sensor_list = []
    for actor in actor_list:
        if actor.parent is not None and actor.parent.type_id == vehicle.type_id:
            sensor_list.append(actor)
    return sensor_list

def get_specific_sensor(world, vehicle, sensor_type):
    sensor_list = get_sensors(world, vehicle)
    for sensor in sensor_list:
        if sensor_type in sensor.type_id:
            return sensor
    return None

class CarlaSensorListMaster(object):
    def __init__(self):
        self.sensor_list = []

    def append(self, sensor, transform, binded):
        sensor_master = CarlaSensorMaster(sensor, transform, binded)
        self.sensor_list.append(sensor_master)

    def destroy(self):
        for sensor_master in self.sensor_list:
            sensor_master.destroy()


class Sensor(object):
    def __init__(self, transform, config):
        self.type_id = 'sensor.camera.rgb'
        self.transform = transform
        self.attributes = dict()
        self.attributes['role_name'] = 'front'
        self.attributes['image_size_x'] = str( config['img_length'] )
        self.attributes['image_size_y'] = str( config['img_width'] )
        self.attributes['fov'] = str( config['fov'] )


class CarlaSensorMaster(object):
    def __init__(self, sensor, transform, binded):
        self.sensor = sensor
        self.transform = transform
        self.raw_data, self.data = None, None

        self.type_id = sensor.type_id
        self.attributes = sensor.attributes

        weak_self = weakref.ref(self)
        if 'lidar' in sensor.type_id:
            self.frame_id = 'lidar/{}'.format(sensor.attributes['role_name'])
            if binded is False:
                self.callback = lambda data: CarlaSensorCallback.lidar(weak_self, data)
                self.sensor.listen(self.callback)
        elif 'camera' in sensor.type_id:
            self.frame_id = 'camera_rgb/{}'.format(sensor.attributes['role_name'])
            if binded is False:
                self.callback = lambda data: CarlaSensorCallback.camera_rgb(weak_self, data)
                self.sensor.listen(self.callback)
        elif 'gnss' in sensor.type_id:
            self.frame_id = 'gnss/{}'.format(sensor.attributes['role_name'])
            if binded is False:
                self.callback = lambda data: CarlaSensorCallback.gnss(weak_self, data)
                self.sensor.listen(self.callback)
        elif 'imu' in sensor.type_id:
            self.frame_id = 'imu/{}'.format(sensor.attributes['role_name'])
            if binded is False:
                self.callback = lambda data: CarlaSensorCallback.imu(weak_self, data)
                self.sensor.listen(self.callback)


    def get_transform(self):
        return self.transform

    """
    def publish(self, data):
        if 'camera' in self.type_id and self.ros_pub is True:
            # data: the result of cv2.imread, np.array
            image = CarlaSensorDataConversion.cv2ImageToSensorImage(data, self.frame_id)
            self.publisher.publish(image)
    

    def get_tf_stamped(self, vehicle_frame_id):
        tf_stamped = TransformStamped()
        tf_stamped.header.stamp = rospy.Time.now()
        tf_stamped.header.frame_id = vehicle_frame_id
        tf_stamped.child_frame_id = self.frame_id
        tf_stamped.transform.translation.x = self.transform.location.x
        tf_stamped.transform.translation.y = self.transform.location.y
        tf_stamped.transform.translation.z = self.transform.location.z
        roll, pitch, yaw = np.deg2rad(self.transform.rotation.roll), np.deg2rad(self.transform.rotation.pitch), np.deg2rad(self.transform.rotation.yaw)
        quaternion = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
        tf_stamped.transform.rotation.x = quaternion[0]
        tf_stamped.transform.rotation.y = quaternion[1]
        tf_stamped.transform.rotation.z = quaternion[2]
        tf_stamped.transform.rotation.w = quaternion[3]
        return tf_stamped
    """

    def destroy(self):
        self.sensor.destroy()

class CarlaSensorCallback(object):
    @staticmethod
    def lidar(weak_self, data):
        # data: carla.LidarMeasurement
        self = weak_self()
        self.raw_data = data
        #self.data = CarlaSensorDataConversion.lidarMeasurementToPointcloud2(data, self.frame_id)
        #print('callback: ' + str(type(self.data)))

    @staticmethod
    def camera_rgb(weak_self, data):
        # data: carla.Image
        self = weak_self()
        self.raw_data = data
        #self.data = CarlaSensorDataConversion.carlaImageToSensorImage(data, self.frame_id)

    @staticmethod
    def gnss(weak_self, data):
        # data: carla.GNSSMeasurement
        self = weak_self()
        self.raw_data = data
        self.data = data

    @staticmethod
    def imu(weak_self, data):
        # data: carla.IMUMeasurement
        self = weak_self()
        self.raw_data = data
        self.data = data