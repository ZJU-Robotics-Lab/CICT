#!/usr/bin/env python3
from __future__ import print_function
# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================
import os
import sys
from os.path import join, dirname
sys.path.insert(0, join(dirname(__file__), '..'))
import simulator
simulator.load('/home/wang/CARLA_0.9.9.4')

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================
import carla
sys.path.append('/home/wang/CARLA_0.9.9.4/PythonAPI/carla')
from agents.navigation.basic_agent import BasicAgent
from simulator import set_weather
from simulator.sensor_manager import SensorManager
from utils.navigator_sim import get_random_destination, get_map, get_nav, get_big_nav, replan, close2dest

import argparse
import random
import re
import copy

from simulator.official_code import DualControl, CameraManager, HUD
import pygame

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')
try:
    import cv2
except ImportError:
    raise RuntimeError('cannot import cv2, make sure python-opencv package is installed')

# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================
global_img = None
global_pcd = None
global_nav = None
global_control = None
global_pos = None
global_vel = None

def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================


class World(object):
    def __init__(self, carla_world, hud):
        self.world = carla_world
        self.hud = hud
        self.player = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
    
        self.restart()
        self.world.on_tick(hud.on_world_tick)

    def restart(self):
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        # Get a blueprint.
        blueprint = random.choice(self.world.get_blueprint_library().filter('vehicle.audi.a2'))
        blueprint.set_attribute('role_name', 'hero')
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        # Spawn the player.
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        while self.player is None:
            spawn_points = self.world.get_map().get_spawn_points()
            spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        # Set up the sensors.
        self.camera_manager = CameraManager(self.player, self.hud)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
    
        sensor_dict = {
        'camera':{
            'transform':carla.Transform(carla.Location(x=0.5, y=0.0, z=2.5)),
            'callback':image_callback,
            },
        'lidar':{
            'transform':carla.Transform(carla.Location(x=0.5, y=0.0, z=2.5)),
            'callback':lidar_callback,
            },
        }

        self.sm = SensorManager(self.world, self.world.get_blueprint_library(), self.player, sensor_dict)
        self.sm.init_all()
        
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def tick(self, clock):
        self.hud.tick(self, clock)

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy(self):
        actors = [
            self.camera_manager.sensor,
            self.player]
        self.sm.close_all()
        for actor in actors:
            if actor is not None:
                actor.destroy()

# ==============================================================================
# -- data collect --------------------------------------------------------------------
# ==============================================================================
def save_data(index):
    global global_img, global_pcd, global_nav, global_control, global_pos, global_vel, global_acceleration, global_angular_velocity
    cv2.imwrite(save_path+'img/'+str(index)+'.png', global_img)
    cv2.imwrite(save_path+'nav/'+str(index)+'.png', global_nav)
    np.save(save_path+'pcd/'+str(index)+'.npy', global_pcd)
    cmd_file.write(index+'\t'+
                   str(global_control.throttle)+'\t'+
                   str(global_control.steer)+'\t'+
                   str(global_control.brake)+'\n')
    pos_file.write(index+'\t'+
                   str(global_pos.location.x)+'\t'+
                   str(global_pos.location.y)+'\t'+
                   str(global_pos.location.z)+'\t'+
                   str(global_pos.rotation.pitch)+'\t'+
                   str(global_pos.rotation.yaw)+'\t'+
                   str(global_pos.rotation.roll)+'\t'+'\n')
    vel_file.write(index+'\t'+
                   str(global_vel.x)+'\t'+
                   str(global_vel.y)+'\t'+
                   str(global_vel.z)+'\t'+'\n')
    acc_file.write(index+'\t'+
                   str(global_acceleration.x)+'\t'+
                   str(global_acceleration.y)+'\t'+
                   str(global_acceleration.z)+'\t'+'\n')
    angular_vel_file.write(index+'\t'+
                   str(global_angular_velocity.x)+'\t'+
                   str(global_angular_velocity.y)+'\t'+
                   str(global_angular_velocity.z)+'\t'+'\n')

def image_callback(data):
    global global_img
    array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8")) 
    array = np.reshape(array, (data.height, data.width, 4)) # RGBA format
    global_img = array
    
def lidar_callback(data):
    global global_pcd
    lidar_data = np.frombuffer(data.raw_data, dtype=np.float32).reshape([-1, 3])
    point_cloud = np.stack([-lidar_data[:,1], -lidar_data[:,0], -lidar_data[:,2]])
    mask = np.where(point_cloud[2] > -2.3)[0]
    point_cloud = point_cloud[:, mask]
    global_pcd = point_cloud
    
# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================

if __name__ == '__main__':
    import time
    # force feedback
    import evdev
    from evdev import ecodes, InputDevice
    device = evdev.list_devices()[0]
    evtdev = InputDevice(device)
    val = 25000 #[0,65535]
    evtdev.write(ecodes.EV_FF, ecodes.FF_AUTOCENTER, val)
    
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1920x1080',
        help='window resolution (default: 1920x1080)')
    argparser.add_argument('-d', '--data', type=int, default=10, help='data index')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    data_index = args.data
    save_path = '/media/wang/DATASET/CARLA_HUMAN/town01/'+str(data_index)+'/'
    
    def mkdir(path):
        os.makedirs(save_path+path, exist_ok=True)
    
    mkdir('')
    mkdir('img/')
    mkdir('pcd/')
    mkdir('nav/')
    mkdir('state/')
    mkdir('cmd/')
    cmd_file = open(save_path+'cmd/cmd.txt', 'w+')
    pos_file = open(save_path+'state/pos.txt', 'w+')
    vel_file = open(save_path+'state/vel.txt', 'w+')
    acc_file = open(save_path+'state/acc.txt', 'w+')
    angular_vel_file = open(save_path+'state/angular_vel.txt', 'w+')
    world = None
    try:
        pygame.init()
        pygame.font.init()
        client = carla.Client(args.host, args.port)
        client.set_timeout(2.0)

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        hud = HUD(args.width, args.height)
        _world = client.load_world('Town01')
        world_map = _world.get_map()
        world = World(_world, hud)
        
        weather = carla.WeatherParameters(
            cloudiness=random.randint(0,80),
            precipitation=0,
            sun_altitude_angle=random.randint(40,90)
        )
        set_weather(_world, weather)
        
        controller = DualControl(world)
        
        vehicle = world.player
        agent = BasicAgent(vehicle, target_speed=31)
        
        spawn_points = world_map.get_spawn_points()
        waypoint_tuple_list = world_map.get_topology()
        origin_map = get_map(waypoint_tuple_list)

        destination = get_random_destination(spawn_points)
        plan_map = replan(agent, destination, copy.deepcopy(origin_map))
        #plan_map.resize((1000, 1000)).show()
        clock = pygame.time.Clock()

        while True:
            if close2dest(vehicle, destination):
                destination = get_random_destination(spawn_points)
                plan_map = replan(agent, destination, copy.deepcopy(origin_map))
            clock.tick_busy_loop(60)
            if controller.parse_events(world, clock):
                break
            if vehicle.is_at_traffic_light():
                traffic_light = vehicle.get_traffic_light()
                if traffic_light.get_state() == carla.TrafficLightState.Red:
                    traffic_light.set_state(carla.TrafficLightState.Green)
            
            nav = get_nav(vehicle, plan_map)
            big_nav = get_big_nav(vehicle, plan_map)
            global_nav = nav
            global_pos = vehicle.get_transform()
            global_vel = vehicle.get_velocity()
            global_control = controller._control
            global_acceleration = vehicle.get_acceleration()
            global_angular_velocity = vehicle.get_angular_velocity()
        
            cv2.imshow('Vision', big_nav)
            cv2.waitKey(10)
            index = str(time.time())
            save_data(index)
            
            world.tick(clock)
            world.render(display)
            pygame.display.flip() 
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
    finally:
        if world is not None:
            world.destroy()
            pygame.quit()