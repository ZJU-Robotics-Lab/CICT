import time
import threading
import can
import os
import logging
import struct
from can import BusABC, Message

CHANNEL = 'COM4'

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
        self, channel, baudrate=115200, timeout=0.1, rtscts=False, *args, **kwargs
    ):
        """ Baud rate of the serial device in bit/s (default 115200). """
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
        byte_msg.append(0xAA)
        # for i in range(0, 4):
        #     byte_msg.append(timestamp[i])
        byte_msg.append(0xc8)
        byte_msg.append(0x01)
        byte_msg.append(0x23)
        # byte_msg.append(msg.dlc)
        # for i in range(0, 4):
            # byte_msg.append(a_id[i])
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

        if rx_byte and ord(rx_byte) == 0xAA:
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


def send_cyclic(bus, msg, stop_event):
    """The loop for sending."""
    print("Start to send a message every 1s")
    start_time = time.time()
    while not stop_event.is_set():
        msg.timestamp = time.time() - start_time
        bus.send(msg)
        print(f"tx: {msg}")
        time.sleep(0.001)
    print("Stopped sending messages")


def receive(bus, stop_event):
    """The loop for receiving."""
    print("Start receiving messages")
    while not stop_event.is_set():
        rx_msg = bus.recv(1)
        if rx_msg is not None:
            print("rx: {}".format(rx_msg))
    print("Stopped receiving messages")


def main():
    """Controles the sender and receiver."""
    # with can.interface.Bus(bustype="serial", channel=CHANNEL) as server:
    with SerialBus(channel=CHANNEL, baudrate=1228800) as server:
            tx_msg = can.Message(
                arbitration_id=0x203,
                data=[0x03, # 03 forward 05 backward 09 break
                      0x00, 0x06, # speed low-high 0~2700 rpm
                      0x40, # acc 0.2~25.5s
                      0x00, # none
                      0x00, 0x05, # ratation 580-1220 low-high
                      0x00 # none
                      ],
            )

            # Thread for sending and receiving messages
            stop_event = threading.Event()
            t_send_cyclic = threading.Thread(
                target=send_cyclic, args=(server, tx_msg, stop_event)
            )
            t_receive = threading.Thread(target=receive, args=(server, stop_event))
            t_receive.start()
            t_send_cyclic.start()

            try:
                while True:
                    time.sleep(0)  # yield
            except KeyboardInterrupt:
                pass  # exit normally

            stop_event.set()
            time.sleep(0.01)

    print("Stopped script")



if __name__ == "__main__":
    main()