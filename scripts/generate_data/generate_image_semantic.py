#!/usr/bin/env python3
from __future__ import print_function
# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================
import os
import sys
import time
from os.path import join, dirname
sys.path.insert(0, join(dirname(__file__), '../..'))
import simulator
simulator.load('/media/jessy104/1TSSD1/CARLA_0.9.9.4')

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================
import carla
sys.path.append('/media/jessy104/1TSSD1/CARLA_0.9.9.4/PythonAPI/carla')
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
from simulator import config

config["camera"]["img_width"]=640
config["camera"]["fov"]=90


global_img = None
global_pcd = None
global_nav = None
global_control = None
global_pos = None
global_vel = None
global_semantic = None

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
            spawn_point = carla.Transform(carla.Location(29.91179847717285,0,0),carla.Rotation(0,0,0))
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
        'semantic':{
            'transform':carla.Transform(carla.Location(x=0.5, y=0.0, z=2.5)),
            'callback':semantic_callback,
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
def save_data(index, weather_condition):
    global global_img, global_semantic, finish_saving
    cv2.imwrite(save_path+'img/'+weather_condition+"/"+str(index)+'.png', global_img)
    cv2.imwrite(save_path+'semantic/'+weather_condition+"/"+str(index)+'.png', global_semantic)
    finish_saving = True


def image_callback(data):
    global global_img, wait_for_update_cam
    array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8")) 
    array = np.reshape(array, (data.height, data.width, 4)) # RGBA format
    global_img = array
    wait_for_update_cam = False

def semantic_callback(data):
    global global_semantic, wait_for_update_semantic
    data.convert(carla.ColorConverter.CityScapesPalette)
    semantic = np.reshape(np.array(data.raw_data),(data.height, data.width, 4))
    global_semantic = deal_semantic_img(semantic)
    # array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8")) 
    # array = np.reshape(array, (data.height, data.width, 4)) # RGBA format
    # global_semantic = array
    wait_for_update_semantic = False
    
def lidar_callback(data):
    global global_pcd
    lidar_data = np.frombuffer(data.raw_data, dtype=np.float32).reshape([-1, 3])
    point_cloud = np.stack([-lidar_data[:,1], -lidar_data[:,0], -lidar_data[:,2]])
    mask = np.where(point_cloud[2] > -2.3)[0]
    point_cloud = point_cloud[:, mask]
    global_pcd = point_cloud


def deal_semantic_img(semantic_img):
    convert_img = semantic_img
    
    OBSTACLE = np.array((0, 0, 0, 255))
    ROAD = np.array((255, 255, 255, 255))

    mask = np.where((semantic_img[:,:,0]!=50) & (semantic_img[:,:,0]!=128))
    convert_img[mask[0], mask[1]] = OBSTACLE

    mask = np.where((semantic_img[:,:,0]==50) | (semantic_img[:,:,0]==128))
    convert_img[mask[0], mask[1]] = ROAD

# TIPS on the color of different object
    # for i in range(IMG_WIDTH):
    #     for j in range(IMG_LENGTH):
    #         pixel = convert_img[i][j]
    #         if pixel[0] == 70: #Building
    #             convert_img[i][j] = OBSTACLE
    #         elif pixel[0] == 190: #Fence
    #             convert_img[i][j] = OBSTACLE
    #         elif pixel[0] == 160: #Other
    #             convert_img[i][j] = OBSTACLE
    #         elif pixel[0] == 220: #Pedestrian
    #             convert_img[i][j] = OBSTACLE
    #         elif pixel[0] == 153: #Pole
    #             convert_img[i][j] = OBSTACLE
    #         elif pixel[0] == 50: #Road line
    #             convert_img[i][j] = ROAD
    #         elif pixel[0] == 128: #Road
    #             convert_img[i][j] = ROAD
    #         elif pixel[0] == 244: #Sidewalk
    #             convert_img[i][j] = ROAD
    #         elif pixel[0] == 35: #Vegetation
    #             convert_img[i][j] = OBSTACLE
    #         elif pixel[0] == 232: #may be Sidewalk
    #             convert_img[i][j] = OBSTACLE
    #         elif pixel[0] == 156: #may be build
    #             convert_img[i][j] = OBSTACLE
    #         elif pixel[0] == 142: #Car
    #             convert_img[i][j] = OBSTACLE
    #         elif pixel[1] == 220: #Traffic sign
    #             convert_img[i][j] = OBSTACLE

    #         ref_pixel = ref_img[i][j]
    #         if ref_pixel[2] > 160 and ref_pixel[0] <30 and ref_pixel[1] <30:
    #             convert_img[i][j] = DEBUG
    # imsave(OUTPUT_PATH + '%06d' % frame + '_seg.png', convert_img)
    return convert_img
    
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
    save_path = '/media/jessy104/1TSSD1//town01/'+str(data_index)+'/'
    
    def mkdir(path):
        os.makedirs(save_path+path, exist_ok=True)
    
    mkdir('')
    mkdir('img/')
    mkdir('pcd/')
    mkdir('nav/')
    mkdir('state/')
    mkdir('cmd/')
    mkdir('semantic/')
    mkdir('semantic/sunny/')
    mkdir('semantic/rainy/')
    mkdir('semantic/foggy/')
    mkdir('semantic/night/')
    mkdir('img/sunny')
    mkdir('img/rainy/')
    mkdir('img/foggy/')
    mkdir('img/night/')
    cmd_file = open(save_path+'cmd/cmd.txt', 'w+')
    pos_file = open(save_path+'state/pos.txt', 'w+')
    vel_file = open(save_path+'state/vel.txt', 'w+')
    acc_file = open(save_path+'state/acc.txt', 'w+')
    angular_vel_file = open(save_path+'state/angular_vel.txt', 'w+')
    world = None
    finish_saving = True
    wait_for_update_cam,  wait_for_update_semantic= True, True
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
            # cloudiness=random.randint(0,80),
            cloudiness=50,
            precipitation=99,
            # sun_altitude_angle=random.randint(40,90)
            fog_density=0,
            wetness=99,
            sun_altitude_angle=90
        )
        # set_weather(_world, weather)
        weather_cycle = ["sunny", "rainy", "foggy", "night"]
        weather_count = 0
        # _world.set_weather(carla.WeatherParameters.HardRainNoon)
        # weather = _world.get_weather()
        
        controller = DualControl(world)
        
        vehicle = world.player
        vehicle.set_transform(carla.Transform(carla.Location(29.91179847717285,0,0),carla.Rotation(0,0,0)))
        agent = BasicAgent(vehicle, target_speed=31)
        
        spawn_points = world_map.get_spawn_points()
        # waypoint_tuple_list = world_map.get_topology()
        # origin_map = get_map(waypoint_tuple_list)

        # destination = get_random_destination(spawn_points)
        # plan_map = replan(agent, destination, copy.deepcopy(origin_map))
        #plan_map.resize((1000, 1000)).show()
        clock = pygame.time.Clock()

        while True: 
            spawn_points = world_map.get_spawn_points()
            print(weather_cycle[weather_count]) 
            this_location = random.choice(spawn_points)
            if weather_cycle[weather_count] == "sunny":
                index = str(time.time())
                vehicle.set_transform(this_location)
                # print("location", this_location)
                weather = carla.WeatherParameters(
                    # cloudiness=random.randint(0,80),
                    cloudiness=20,
                    precipitation=0,
                    # sun_altitude_angle=random.randint(40,90)
                    fog_density=0,
                    wetness=99,
                    sun_altitude_angle=90
                )
                   
                weather_count += 1
            elif weather_cycle[weather_count] == "rainy":
                # rainy_location = this_location+carla.Transform(carla.Location(10.0,5.0,0.0),carla.Rotation(0,0,0))
                # vehicle.set_transform(rainy_location)
                # print("new location",this_location)                
                weather = carla.WeatherParameters(
                    # cloudiness=random.randint(0,80),
                    cloudiness=20,
                    precipitation=99,
                    # sun_altitude_angle=random.randint(40,90)
                    fog_density=0,
                    wetness=99,
                    sun_altitude_angle=30
                )
                
                weather_count += 1  
            elif weather_cycle[weather_count] == "foggy":
                weather = carla.WeatherParameters(
                    # cloudiness=random.randint(0,80),
                    cloudiness=20,
                    precipitation=0,
                    # sun_altitude_angle=random.randint(40,90)
                    fog_density=60,
                    wetness=99,
                    sun_altitude_angle=90
                )
                
                weather_count += 1       
            elif weather_cycle[weather_count] == "night":
                weather = carla.WeatherParameters(
                    # cloudiness=random.randint(0,80),
                    cloudiness=20,
                    precipitation=0,
                    # sun_altitude_angle=random.randint(40,90)
                    fog_density=0,
                    wetness=0,
                    sun_altitude_angle=0
                )
                
                weather_count = 0 

            set_weather(_world, weather) 

            wait_for_update_cam,  wait_for_update_semantic= True, True

            while wait_for_update_cam and wait_for_update_semantic:
                time.sleep(0.5)               
                                
            # if close2dest(vehicle, destination):
            #     destination = get_random_destination(spawn_points)
            #     plan_map = replan(agent, destination, copy.deepcopy(origin_map))
            # clock.tick_busy_loop(60)
            # if controller.parse_events(world, clock):
            #     break
            # if vehicle.is_at_traffic_light():
            #     traffic_light = vehicle.get_traffic_light()
            #     if traffic_light.get_state() == carla.TrafficLightState.Red:
            #         traffic_light.set_state(carla.TrafficLightState.Green)
            
            # nav = get_nav(vehicle, plan_map)
            # big_nav = get_big_nav(vehicle, plan_map)
            # global_nav = nav
            # global_pos = vehicle.get_transform()
            # global_vel = vehicle.get_velocity()
            # global_control = controller._control
            # global_acceleration = vehicle.get_acceleration()
            # global_angular_velocity = vehicle.get_angular_velocity()
        
            # cv2.imshow('Vision', big_nav)
            # cv2.waitKey(10)
            
            # save_data(index, weather_count[weather_count])
            finish_saving = False
            save_data(index, weather_cycle[weather_count])
            
            world.tick(clock)
            world.render(display)
            pygame.display.flip() 
            while not finish_saving:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
    finally:
        if world is not None:
            world.destroy()
            pygame.quit()