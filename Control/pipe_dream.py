import socket
import threading
import time
from pynput import keyboard

# UDP settings
UDP_IP = "192.168.4.1"
UDP_PORT = 4210

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

held_keys = set()
send_interval = 0.05  # message period
running = True  # Flag to stop the thread gracefully

# key mapping
key_map = {
    keyboard.Key.up: "UP",
    keyboard.Key.down: "DOWN",
    keyboard.Key.left: "LEFT",
    keyboard.Key.right: "RIGHT"
}

# Background thread to send held keys repeatedly
def send_loop():
    while running:
        for key in held_keys:
            message = key_map.get(key)
            if message:
                print(message)
                sock.sendto(message.encode(), (UDP_IP, UDP_PORT))
        time.sleep(send_interval)

# Key press/release handlers
def on_press(key):
    if key in key_map:
        held_keys.add(key)

def on_release(key):
    if key in held_keys:
        held_keys.remove(key)
        # Send STOP message a few times to make sure it is received
        for _ in range(3):
            sock.sendto(b"STOP", (UDP_IP, UDP_PORT))
            time.sleep(0.05)

# Start background sending thread
send_thread = threading.Thread(target=send_loop)
send_thread.start()

# Start listening to keyboard
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

print("Listening...")

try:
    while True:
        time.sleep(0.05)
except KeyboardInterrupt:
    print("Exiting program...")
    running = False
    listener.stop()
    send_thread.join()

        