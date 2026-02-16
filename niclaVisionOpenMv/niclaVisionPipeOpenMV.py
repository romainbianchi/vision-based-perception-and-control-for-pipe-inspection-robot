import network
import socket
import time
import struct
import sensor
import pyb
import gc
import asyncio
import uselect

from machine import I2C
from vl53l1x import VL53L1X

from lsm6dsox import LSM6DSOX
from machine import Pin
from machine import SPI

# lsm = LSM6DSOX(SPI(5), cs=Pin("PF6", Pin.OUT_PP, Pin.PULL_UP))
lsm = LSM6DSOX(
    SPI(5),
    cs=Pin("PF6", Pin.OUT_PP, Pin.PULL_UP),
    gyro_odr     = 52,     # 52 Hz → dt ≈ 19 ms
    accel_odr    = 52,
    gyro_scale   = 250,    #  ±250 °/s    ( 8.75 mdps / LSB )
    accel_scale  = 4 
)

# WiFi Credentials
SSID = "pipedreamAP"
PASSWORD = "pipe_dream"

# Server Details
SERVER_IP = "192.168.4.2"
SERVER_CAMERA_PORT = 12345
SERVER_IMU_PORT = 12346

IMAGE_INTERVAL = 100
IMU_INTERVAL = 20

COMPRESSION_QUALITY = 70

# Initialize WiFi
wlan = network.WLAN(network.STA_IF)
wlan.active(True)
wlan.connect(SSID, PASSWORD)

while not wlan.isconnected():
    print("Trying to connect to WiFi...")
    time.sleep(1)

print("Connected to WiFi:", wlan.ifconfig())

# Connect to Server
camera_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
camera_client.connect((SERVER_IP, SERVER_CAMERA_PORT))
imu_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
imu_client.connect((SERVER_IP, SERVER_IMU_PORT))
print('connected to server')


# Initialize camera
sensor.reset()
sensor.set_pixformat(sensor.RGB565)  # or sensor.GRAYSCALE
sensor.set_framesize(sensor.QVGA)  # 320x240
sensor.skip_frames(time=2000)
clock = time.clock()
print("Camera ready")

# Initialize ToF
tof = VL53L1X(I2C(2))


async def send_image():

    next_frame = True

    # Non-blocking ACK check
    poller = uselect.poll()
    poller.register(camera_client, uselect.POLLIN)


    while True:
        try:

            # Check if server is ready for next frame
            if not next_frame:
                if poller.poll(0):  # Immediate return
                    ack = camera_client.recv(1)
                    if ack == b'0':
                        next_frame = True
                        asyncio.sleep_ms(10)
                        continue
                    else:
                        print("Unexpected ACK:", ack)
                        break

                await asyncio.sleep_ms(50)
                continue

            # Capture and prepare frame
            frame = sensor.snapshot().compress(quality=COMPRESSION_QUALITY)
            img_bytes = frame.bytearray()
            gc.collect()
            frame_size = len(img_bytes)

            next_frame = False

            try:
                # Send frame size header
                camera_client.sendall(struct.pack('<I', frame_size))

                # Send frame data in chunks
                CHUNK_SIZE = 1024
                bytes_sent = 0
                while bytes_sent < frame_size:
                    chunk = img_bytes[bytes_sent:bytes_sent+CHUNK_SIZE]
                    sent = camera_client.send(chunk)
                    bytes_sent += sent

            except Exception as e:
                print("Frame send error:", e)

        except Exception as e:
            print("Capture error:", e)

        await asyncio.sleep_ms(IMAGE_INTERVAL)


async def send_IMU():
    while True:
        try:
            timestamp = time.ticks_us()

            # Read IMU data once to ensure consistency
            accel_x, accel_y, accel_z = lsm.accel()
            gyro_x, gyro_y, gyro_z = lsm.gyro()

            # Convert to m/s^2
            accel_x *= 9.81
            accel_y *= 9.81
            accel_z *= 9.81

            # Pack all data into a single message
            data = struct.pack('<QB6f',
                               timestamp,
                              0x49,  # IMU message identifier
                              accel_x, accel_y, accel_z,
                              gyro_x, gyro_y, gyro_z)

            # Send the complete message
            imu_client.sendall(data)

            # send ToF value
            dist = tof.read()
            imu_client.sendall(struct.pack('<H', dist))

        except Exception as e:
            print("IMU Send error:", e)

        await asyncio.sleep_ms(IMU_INTERVAL)


async def main():
    asyncio.create_task(send_image())
    asyncio.create_task(send_IMU())
    while True:
        await asyncio.sleep(1)

asyncio.run(main())
