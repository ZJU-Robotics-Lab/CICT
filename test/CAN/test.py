import time
import threading

import can

CHANNEL = 'COM5'

def send_cyclic(bus, msg, stop_event):
    """The loop for sending."""
    print("Start to send a message every 1s")
    start_time = time.time()
    while not stop_event.is_set():
        msg.timestamp = time.time() - start_time
        bus.send(msg)
        print(f"tx: {msg}")
        time.sleep(0.1)
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
    with can.interface.Bus(bustype="serial", channel=CHANNEL) as server:
        #with can.interface.Bus(bustype="serial", channel="COM5") as client:

            tx_msg = can.Message(
                arbitration_id=0x203,
                data=[0x03, # 03 forward 05 backward 09 break
                      0x00, 0x80, # speed low-high 0~2700 rpm
                      0x80, # acc 0.2~25.5s
                      0x00, # none
                      0x00, 0x00, # ratation 580-1220 low-high
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
            time.sleep(0.5)

    print("Stopped script")


if __name__ == "__main__":
    main()