#!/usr/bin/env python

import numpy as np

def pi2pi(theta, theta0=0.0):
	# print('pi2pi: '+str(theta)+'----'+str(theta0))
	while(theta > np.pi + theta0):
		theta = theta - 2.0 * np.pi
	while(theta < -np.pi + theta0):
		theta = theta + 2.0 * np.pi
	return theta


# def np_dot(args):
# 	res = args[0]
# 	for arg in args[1:]:
# 		res = np.dot(res, arg)
# 	return res


def np_dot(*args):
	res = args[0]
	for arg in args[1:]:
		res = np.dot(res, arg)
	return res




# import rospy
# import tf
# from geometry_msgs.msg import PoseStamped
# def waypoint_to_pose(waypoint):
# 	# waypoint: carla.Waypoint
# 	pose = PoseStamped()
# 	pose.header.stamp = rospy.Time.now()
# 	pose.header.frame_id = 'map'
# 	pose.pose.position.x = waypoint.transform.location.x
# 	pose.pose.position.y = waypoint.transform.location.y
# 	pose.pose.position.z = waypoint.transform.location.z
# 	pose.pose.position.z = 0

# 	roll_rad = np.deg2rad(waypoint.transform.rotation.roll)
# 	pitch_rad= np.deg2rad(waypoint.transform.rotation.pitch)
# 	yaw_rad  = np.deg2rad(waypoint.transform.rotation.yaw)
# 	quaternion = tf.transformations.quaternion_from_euler(
# 		roll_rad, pitch_rad, yaw_rad)
# 	pose.pose.orientation.x = quaternion[0]
# 	pose.pose.orientation.y = quaternion[1]
# 	pose.pose.orientation.z = quaternion[2]
# 	pose.pose.orientation.w = quaternion[3]
# 	return pose


# from carla_msgs.msg_wrap import CarlaWaypointWrap

# def waypoint_to_msg(waypoint, reference=False, step=None):
# 	# convert carla.Waypoint to carla_msgs/CarlaWaypoint
# 	pose = waypoint_to_pose(waypoint)
# 	return CarlaWaypointWrap(pose, reference=reference, step=step)



# def pose_to_waypoint(pose, reference=False, step=None):
# 	return CarlaWaypointWrap(pose, reference=reference, step=step)







import matplotlib.pyplot as plt

def plot_arrow_2D(generalized_pose, length=1.0, width=0.5, fc="r", ec="k"):  # pragma: no cover
    """
    	generalized_pose: include carla_msgs/CarlaWaypoint, carla_msgs/CarlaState
    """
    x, y, theta = generalized_pose.x, generalized_pose.y, generalized_pose.theta
    plt.arrow(x, y, length * np.cos(theta), length * np.sin(theta),
              fc=fc, ec=ec, head_width=width, head_length=width)
    # plt.plot(x, y, linewidth=10, markersize=5)
    # plt.plot(0, 0)