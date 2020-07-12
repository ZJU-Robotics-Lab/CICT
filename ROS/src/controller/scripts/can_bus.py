#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import threading
import platform
import struct
import serial
import sys
if sys.version > '3':
	from queue import Queue
else:
	from Queue import Queue
from can import BusABC, Message

sys_type = platform.system()
CHANNEL = 'COM4' if sys_type == 'Windows' else '/dev/ttyUSB0'
BAUD_RATE = 115200
SEND_ID = 0x203
RECEIVE_ID_LOW = 0x183
RECEIVE_ID_HIGH = 0x283
RECEIVE_ID_ROTATION = 0x279


class SerialBus(BusABC):
    def __init__(
        self, channel, baudrate=BAUD_RATE, timeout=1, rtscts=False, *args, **kwargs
    ):
        """ Baud rate of the serial device in bit/s (default BAUD_RATE). """
        if not channel:
            raise ValueError("Must specify a serial port.")

        self.channel_info = "Serial interface: " + channel
        self.ser = serial.serial_for_url(
            channel, baudrate=baudrate, timeout=timeout, rtscts=rtscts
        )
        print('channelchannelchannelchannelchannel', channel)
        super(SerialBus, self).__init__(channel=channel, *args, **kwargs)

    def shutdown(self):
        self.ser.close()

    def send(self, msg):
        byte_msg = bytearray()
        byte_msg.append(0xaa)
        byte_msg.append(0xc8)
        byte_msg.append(msg.arbitration_id & 0x00ff) # arbitration_id low 
        byte_msg.append(msg.arbitration_id >> 8) # arbitration_id high
        for i in range(msg.dlc):
            byte_msg.append(msg.data[i])
            # print(hex(msg.data[i]))
        # print("\n")
        byte_msg.append(0x55)
        self.ser.write(byte_msg)

    def recv(self):
        try:
            # ser.read can return an empty string
            # or raise a SerialException
            rx_bytes = self.ser.read(13)
        except serial.SerialException:
            return None
        
        if rx_bytes and struct.unpack_from("B",rx_bytes, offset = 0)[0] == 0xaa:
                # print(hex(struct.unpack_from("B",rx_bytes, offset = i)[0]))
            # print("\n")
            config = struct.unpack_from("B", rx_bytes, offset = 1)[0]
            if config & 0x10:
                print("Oh shit!!! This is remote frame!!!")
            if config & 0x20:
                arb_id_length = 4
            else:
                arb_id_length = 2

            arb_id = 0
            for i in range(arb_id_length):
                arb_id += struct.unpack_from("B", rx_bytes, offset = i + 2)[0] << (i * 8)
            dlc = config & 0x0f
            data = [struct.unpack_from("B", rx_bytes, offset = i+2+arb_id_length)[0] for i in range(dlc)]
            rxd_byte = struct.unpack_from("B", rx_bytes, offset = 2+arb_id_length+dlc)[0]
            if rxd_byte == 0x55:
                # received message data okay
                msg = Message(
                    arbitration_id = arb_id,
                    dlc = dlc,
                    data = data,
                )
                return msg

        else:
            return None


class Controller:
    def __init__(self, channel=CHANNEL, baudrate=BAUD_RATE, send_id = SEND_ID):
        self.bus = SerialBus(
            channel = channel, 
            baudrate = baudrate
        )
        self.send_id = send_id
        self.max_speed = 2000
        self.max_rotation = 700
        self.acc_time = 3
        self.raw_rotation = 0

        # message read from car
        self.cur_motor_pwm_speed = 0        # current motor speed by pwm 0 ~ 2700
        self.cur_rotation = 0               # current rotation
        self.cur_rot_error = False          # rotation EPS error
        self.cur_ctr_error = False          # controller error
        self.cur_battery_temperature = 0    
        self.cur_ctr_emergenry_stop = False # controller has emergentry stopped
        self.cur_ctr_auto = False           # under program's control
        self.cur_battery_power = 100        # 0 ~ 100%
        self.cur_motor_current = 0          # current of motor
        self.cur_speed = 0                  # current speed km/h
        
        self.sensor_data = Queue(100)
        self.update_queue()
        self.has_reversed = False
        self.stop_send = threading.Event()
        self.stop_receive = threading.Event()
        self.send_cyclic = threading.Thread(
                target=self.send, 
                args=()
        )
        self.receive = threading.Thread(
            target=self.receive, 
            args=()
        )
        self.cmd_data = [
                0x09,      # 03 forward 05 backward 09 break
                0x00, 0x00, # speed low-high 0~2700 rpm
                0x00,       # acc 0.2~25.5s
                0x00,       # none
                0x00, 0x00, # ratation 580-1220 low-high
                0x00       # none
                ]
        self.rx_data_low = [
                0x00, 0x00, # low-high: motor speed by pwm 0~2700 rpm 
                0x00, 0x00, # low-high: ratation 580-1220
                0x00,       # error msg: bit 0 is rotation EPS error, bit 1 is controller error
                0x00,       # battery temperature
                0x00,       # bit 1: emergentry stop status
                0x00        # battery power 0~100%
                ]
        self.rx_data_high = [
                0x00, 0x00, # motor current *10 mA
                0x00,       # current car speed km/h
                0x00, 0x00, 0x00, 0x00, 0x00        # none
                ]
        self.rx_data_rotation = [
                0x00, 0x00, # car rotation
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00        # none
                ]

        self.set_acc_time(self.acc_time)
        self.set_forward()

    def start(self):
        self.stop_send.clear()
        self.stop_receive.clear()
        self.send_cyclic.start()
        self.receive.start()

    def stop_event(self):
        self.stop_send.set()
        self.stop_receive.set()
        self.bus.shutdown()

    def set_max_speed(self, max_speed):
        max_speed = min(2700, max(0, max_speed))
        self.max_speed = max_speed

    # acc 0.2~25.5s --> 2-255
    def set_acc_time(self, acc_time):
        acc_time = min(25.5, max(0.2, acc_time))
        self.acc_time = acc_time
        acc_time = int(10*acc_time)
        self.cmd_data[3] = acc_time & 0xff
    
    def get_acc_time(self):
        return self.acc_time
    
    # input (0, 1)
    def set_speed(self, speed):
        speed = min(1.0, max(speed, 0.0)) 
        speed = int(self.max_speed * speed)
        self.cmd_data[1] = speed & 0xff
        self.cmd_data[2] = (speed & 0xff00) >> 8

    def set_forward(self):
        if self.cmd_data[0] == 0x05:
            self.has_reversed = True
        self.cmd_data[0] = 0x03

    def set_backward(self):
        if self.cmd_data[0] == 0x03:
            self.has_reversed = True
        self.cmd_data[0] = 0x05

    def set_stop(self):
        self.cmd_data[0] = 0x09
        self.has_reversed = False

    # input (-1, 1)
    def set_rotation(self, rotation):
        rotation = max(-1.0, min(rotation, 1.0))
        rotation = -int(self.max_rotation*rotation)
        self.raw_rotation = rotation
        symbol = 0
        if(rotation < 0):
            symbol = 1
            rotation = -rotation
            rotation = (rotation ^ 0xffff) + 1
        self.cmd_data[5] = rotation & 0xff
        self.cmd_data[6] = ((rotation & 0xff00) >> 8) | (symbol << 7)

    def get_max_speed(self):
        return self.max_speed
        
    def get_max_rotation(self):
        return self.max_rotation                       

    def get_cmd_acc_time(self):
        return self.acc_time

    def get_cmd_speed(self):
        return self.cmd_data[1] + self.cmd_data[2] << 8

    def get_cmd_rotation(self):
        return self.raw_rotation

    def get_cmd_reversed(self):
        return self.has_reversed

    def get_cur_motor_pwm_speed(self):
        return self.cur_motor_pwm_speed

    def get_cur_rotation(self):
        return self.cur_rotation

    def get_cur_rot_error(self):
        return self.cur_rot_error

    def get_cur_ctr_error(self):
        return self.cur_ctr_error

    def get_cur_battery_temperature(self):
        return self.cur_battery_temperature

    def get_cur_ctr_emergenry_stop(self):
        return self.cur_ctr_emergenry_stop

    def get_cur_ctr_auto(self):
        return self.cur_ctr_auto

    def get_cur_battery_power(self):
        return self.cur_battery_power

    def get_cur_motor_current(self):
        return self.cur_motor_current

    def get_cur_speed(self):
        return self.cur_speed

    def get_sensor_dGPSata(self):
        return self.sensor_data

    def unpack(self):
        self.cur_motor_pwm_speed = (self.rx_data_low[1] << 8) + self.rx_data_low[0]  
        self.cur_rotation = self.rx_data_rotation[0] + ((self.rx_data_rotation[1] & 0x7f) << 8)
        if bool(self.rx_data_rotation[1] & 0x80):
            self.cur_rotation -= 0x8000
        self.cur_rotation = -self.cur_rotation                               
        self.cur_rot_error = bool(self.rx_data_low[4] & 0x01)          
        self.cur_ctr_error = bool(self.rx_data_low[4] & 0x02)          
        self.cur_battery_temperature = self.rx_data_low[5]    
        self.cur_ctr_emergenry_stop = bool(self.rx_data_low[6] & 0x01)
        self.cur_ctr_auto = bool(self.rx_data_low[6] & 0x02)           
        self.cur_battery_power = self.rx_data_low[7]        
        self.cur_motor_current = self.rx_data_high[1] << 8 + self.rx_data_high[0]          
        self.cur_speed = self.rx_data_high[2]
        self.update_queue()                  
    
    def update_queue(self):
        if not self.sensor_data.empty():
            self.sensor_data.queue.clear()
        self.sensor_data.put(self.cur_motor_pwm_speed)
        self.sensor_data.put(self.cur_rotation)
        self.sensor_data.put(self.cur_rotation)
        self.sensor_data.put(self.cur_rot_error)
        self.sensor_data.put(self.cur_ctr_error)
        self.sensor_data.put(self.cur_battery_temperature)
        self.sensor_data.put(self.cur_ctr_emergenry_stop)
        self.sensor_data.put(self.cur_ctr_auto)
        self.sensor_data.put(self.cur_battery_power)
        self.sensor_data.put(self.cur_motor_current)
        self.sensor_data.put(self.cur_speed)

    def send(self):
        """The loop for sending."""
        print("Start to send messages")
        start_time = time.time()
        while not self.stop_send.is_set():
            msg = Message(
                arbitration_id = self.send_id,
                data = self.cmd_data
            )
            msg.timestamp = time.time() - start_time
            self.bus.send(msg)
            # print(f"tx: {msg}")
            time.sleep(0.001)
        print("Stopped sending messages")

    def receive(self):
        print("Start receiving messages")
        while not self.stop_receive.is_set():
            rx_msg = self.bus.recv()
            if rx_msg is not None:
                if rx_msg.arbitration_id == RECEIVE_ID_LOW:
                    # print("low receive")
                    self.rx_data_low = rx_msg.data
                    self.unpack()
                elif rx_msg.arbitration_id == RECEIVE_ID_HIGH:
                    # print("high receive")
                    self.rx_data_high = rx_msg.data
                    self.unpack()
                elif rx_msg.arbitration_id == RECEIVE_ID_ROTATION:
                    self.rx_data_rotation = rx_msg.data
                    self.unpack()
            # if rx_msg is not None:
                # print("rx: {}".format(rx_msg))
        print("Stopped receiving messages")


if __name__ == "__main__":
    ctrl = Controller()
    ctrl.start()
    ctrl.set_forward()
    ctrl.set_speed(0)
    # ctrl.set_max_rotation(500)
    # ctrl.set_speed(0)
    ctrl.set_acc_time(1.0)
    # print('rotation start',ctrl.get_cur_rotation())
    ctrl.set_rotation(-0.9)
    # print('rotation changed',ctrl.get_cur_rotation())
    time.sleep(5)
    ctrl.stop_event()