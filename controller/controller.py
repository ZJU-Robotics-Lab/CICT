#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import threading
import platform
import struct
import serial

import can
from can import BusABC, Message

sys_type = platform.system()
CHANNEL = 'COM5' if sys_type == 'Windows' else '/dev/ttyUSB0'
BAUD_RATE = 1228800
ID = 0x203

class SerialBus(BusABC):
    def __init__(
        self, channel, baudrate=BAUD_RATE, timeout=0.1, rtscts=False, *args, **kwargs
    ):
        """ Baud rate of the serial device in bit/s (default BAUD_RATE). """
        if not channel:
            raise ValueError("Must specify a serial port.")

        self.channel_info = "Serial interface: " + channel
        self.ser = serial.serial_for_url(
            channel, baudrate=baudrate, timeout=timeout, rtscts=rtscts
        )
        super().__init__(channel=channel, *args, **kwargs)

    def shutdown(self):
        self.ser.close()

    def send(self, msg):

        byte_msg = bytearray()
        byte_msg.append(0xaa)
        byte_msg.append(0xc8)
        byte_msg.append(msg.arbitration_id & 0x00ff) # arbitration_id low 
        byte_msg.append(msg.arbitration_id >> 8) # arbitration_id high
        for i in range(0, msg.dlc):
            byte_msg.append(msg.data[i])
        byte_msg.append(0x55)
        self.ser.write(byte_msg)

    def recv(self):
        try:
            # ser.read can return an empty string
            # or raise a SerialException
            rx_byte = self.ser.read()
        except serial.SerialException:
            return None, False

        if rx_byte and ord(rx_byte) == 0xaa:
            config = self.ser.read()

            if config & 0x10:
                print("Oh shit!!! This is remote frame!!!")
            if config & 0x20:
                arb_id_length = 4
            else:
                arb_id_length = 2

            arb_id = 0
            for i in range(arb_id_length):
                arb_id += self.ser.read() << (i * 8)
            dlc = config & 0x0f
            data = self.ser.read(dlc)
            rxd_byte = ord(self.ser.read())
            if rxd_byte == 0x55:
                # received message data okay
                msg = Message(
                    arbitration_id = arb_id,
                    dlc = dlc,
                    data = data,
                )
                return msg, False

        else:
            return None, False


class Controller:
    def __init__(self, channel=CHANNEL, baudrate=BAUD_RATE):
        self.bus = SerialBus(
            channel = channel, 
            baudrate = baudrate
        )
        self.max_speed = 1000
        self.max_rotation = 580
        self.acc_time = 10
        
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
        
    def set_max_rotation(self, max_rotation):
        self.max_rotation = min(580, max(0, max_rotation))
        
    # acc 0.2~25.5s
    def set_acc_time(self, acc_time):
        acc_time = min(30, max(5, acc_time))
        self.acc_time = acc_time
        acc_time = int(acc_time)
        self.cmd_data[3] = acc_time & 0xff
        
    def get_max_speed(self):
        return self.max_speed
        
    def get_max_rotation(self):
        return self.max_rotation
    
    def get_acc_time(self):
        return self.acc_time
    
    def apply_config(self):
        self.stop_send.set()
        if self.has_reversed:
            self.has_reversed = False
            self.stop_now(1) # stop for 1 second if the dir reversed

        while self.send_cyclic.is_alive():
            pass
        self.stop_send.clear()
        self.send_cyclic = threading.Thread(
            target=self.send, 
            args=()
        )
        self.send_cyclic.start()


    # input (0, 1)
    def set_speed(self, speed):
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

    # input (-1, 1)
    def set_rotation(self, rotation):
        rotation = int(self.max_rotation * rotation)
        self.cmd_data[5] = rotation & 0xff
        self.cmd_data[6] = (rotation & 0xff00) >> 8

    def stop_now(self, time):
        data = [0x09,      # 03 forward 05 backward 09 break
                0x00, 0x00, # speed low-high 0~2700 rpm
                0x00,       # acc 0.2~25.5s
                0x00,       # none
                0x00, 0x00, # ratation 580-1220 low-high
                0x00       # none
                ]
        tx_msg = can.Message(
                arbitration_id = ID,
                data = data
                )
        self.stop_send.set()
        while self.send_cyclic.is_alive():
            pass
        self.stop_send.clear()
        self.send_cyclic = threading.Thread(
                target=self.send, 
                args=(tx_msg,)
        )
        
        self.send_cyclic.start()
        time.sleep(time)
        self.stop_send.set()
        while self.send_cyclic.is_alive():
            pass
        self.stop_send.clear()



    def send(self):
        """The loop for sending."""
        print("Start to send messages")
        start_time = time.time()
        while not self.stop_send.is_set():
            msg = can.Message(
                arbitration_id = ID,
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
            # if rx_msg is not None:
                # print("rx: {}".format(rx_msg))
        print("Stopped receiving messages")




if __name__ == "__main__":
    ctrl = Controller()
    ctrl.start()
    ctrl.set_speed(0)
    ctrl.set_acc_time(0)
    ctrl.set_rotation(-580)
    ctrl.set_forward()
    time.sleep(5)
    ctrl.stop_event()