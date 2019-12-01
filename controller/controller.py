import time
import threading
import can
import os
import logging
import struct
from can import BusABC, Message

CHANNEL = 'COM4'
BAUD_RATE = 1228800
ID = 0x203

logger = logging.getLogger("can.serial")

try:
    import serial
except ImportError:
    logger.warning(
        "You won't be able to use the serial can backend without "
        "the serial module installed!"
    )
    serial = None


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

    def send(self, msg, timeout=None):

        try:
            timestamp = struct.pack("<I", int(msg.timestamp * 1000))
        except struct.error:
            raise ValueError("Timestamp is out of range")
        try:
            a_id = struct.pack("<I", msg.arbitration_id)
        except struct.error:
            raise ValueError("Arbitration Id is out of range")
        byte_msg = bytearray()
        byte_msg.append(0xaa)
        byte_msg.append(0xc8)
        byte_msg.append(msg.arbitration_id & 0x00ff) # arbitration_id low 
        byte_msg.append(msg.arbitration_id >> 8) # arbitration_id high
        for i in range(0, msg.dlc):
            byte_msg.append(msg.data[i])
        byte_msg.append(0x55)
        self.ser.write(byte_msg)

    def recv(self, timeout):
        try:
            # ser.read can return an empty string
            # or raise a SerialException
            rx_byte = self.ser.read()
        except serial.SerialException:
            return None, False

        if rx_byte and ord(rx_byte) == 0xaa:
            s = bytearray(self.ser.read(4))
            timestamp = (struct.unpack("<I", s))[0]
            dlc = ord(self.ser.read())

            s = bytearray(self.ser.read(4))
            arb_id = (struct.unpack("<I", s))[0]
            data = self.ser.read(dlc)
            rxd_byte = ord(self.ser.read())
            if rxd_byte == 0xBB:
                # received message data okay
                msg = Message(
                    timestamp=timestamp / 1000,
                    arbitration_id=arb_id,
                    dlc=dlc,
                    data=data,
                )
                return msg, False

        else:
            return None, False

    def fileno(self):
        if hasattr(self.ser, "fileno"):
            return self.ser.fileno()
        return -1


class Controller:
    def __init__(self, ctr_channel=CHANNEL, ctr_baudrate=BAUD_RATE):
        self.bus = SerialBus(
            channel = ctr_channel, 
            baudrate = ctr_baudrate
        )
        self.cmd_data = [0x09,      # 03 forward 05 backward 09 break
                        0x00, 0x00, # speed low-high 0~2700 rpm
                        0x00,       # acc 0.2~25.5s
                        0x00,       # none
                        0x00, 0x00, # ratation 580-1220 low-high
                        0x00       # none
                        ]
        tx_msg = can.Message(
            arbitration_id = ID,
            data = self.cmd_data
        )
        self.has_reversed = False
        self.stop_send = threading.Event()
        self.stop_receive = threading.Event()
        self.t_send_cyclic = threading.Thread(
                target=send_cyclic, 
                args=(tx_msg)
        )
        self.t_receive = threading.Thread(
            target=receive, 
            args=()
        )

    def start(self):
        self.t_send_cyclic.start()
        self.t_receive.start()

    def stop_event(self):
        self.stop_send.set()
        self.stop_receive.set()

    def apply_config(slef):
        self.stop_send.set()
        if self.has_reversed:
            self.has_reversed = False
            stop_now(1) # stop for 1 second if the dir reversed
        tx_msg = can.Message(
            arbitration_id = ID,
            data = data
        )
        self.t_send_cyclic = threading.Thread(
            target=send_cyclic, 
            args=(tx_msg)
        )
        self.stop_send.clear()
        self.t_send_cyclic.start()


# speed low-high 0~2700 rpm 
    def set_speed(self, speed):
        self.cmd_data[1] = speed & 0xff
        self.cmd_data[2] = (speed & 0xff00) >> 8

# acc 0.2~25.5s
    def set_acc_time(self, acc_time):
        self.cmd_data[3] = acc_time & 0xff

# 03 forward 05 backward 09 break
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

# ratation 580-1220 low-high
    def set_rotation(self, rotation):
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
        self.t_send_cyclic = threading.Thread(
                target=send_cyclic, 
                args=(tx_msg)
        )
        self.stop_send.clear()
        self.t_send_cyclic.start()
        time.sleep(time)
        self.stop_send.set()



    def send_cyclic(self, msg):
        """The loop for sending."""
        print("Start to send messages")
        start_time = time.time()
        while not self.stop_send.is_set():
            msg.timestamp = time.time() - start_time
            bus.send(msg)
            print(f"tx: {msg}")
            time.sleep(0.001)
        print("Stopped sending messages")

    def receive(self):
        """The loop for receiving."""
        print("Start receiving messages")
        while not self.stop_receive.is_set():
            rx_msg = bus.recv(1)
            if rx_msg is not None:
                print("rx: {}".format(rx_msg))
        print("Stopped receiving messages")




if __name__ == "__main__":
    pass