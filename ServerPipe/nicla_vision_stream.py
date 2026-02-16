import socket
import struct
import cv2
import time
import numpy as np
import os

class NiclaVisionStreamer:

    def __init__(self, frame_width, frame_height, byte_per_pixel):

        self.frame_width = frame_width
        self.frame_height = frame_height
        self.byte_per_pixel = byte_per_pixel

        self.frame_size = frame_width * frame_height * byte_per_pixel


    def read_frame(self, client_socket):
        # Read 4-byte header (frame size)
        header = client_socket.recv(4)
        if len(header) != 4:
            return None
        
        frame_size = int.from_bytes(header, byteorder='little')
        
        # Read the entire frame at once
        frame_data = b""
        while len(frame_data) < frame_size:
            chunk = client_socket.recv(frame_size - len(frame_data))
            if not chunk:
                return None
            frame_data += chunk

        client_socket.sendall(b'0')
        
        return frame_data
    

    def read_ToF_value(self, client_socket):
        msg = client_socket.recv(2)
        distance = int.from_bytes(msg, byteorder='little')
        return distance
    

    def read_imu_data(self, client_socket):

        data = client_socket.recv(33)
        if not data or len(data) != 33:
            return None
        
        imu_data = {}
            
        # Unpack and process IMU data
        timestamp, prefix, ax, ay, az, gx, gy, gz = struct.unpack('<QB6f', data)
        imu_data["accel"] = np.array([ax, ay, az])
        imu_data["gyro"] = np.array([gx, gy, gz])

        # Save IMU data to a .txt file
        # with open("imu_data.txt", "a") as f:
        #     f.write(f"{imu_data['accel']}, {imu_data['gyro']}\n")

        return imu_data, timestamp


    def process_compressed_rgb565_frame(self, jpeg_data):
        
        # Convert JPEG byte stream to numpy array
        buffer = np.frombuffer(jpeg_data, dtype=np.uint8)

        # Decode JPEG to BGR image
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

        if image is None:
            print("Failed to decode JPEG image")
            print("image is none")
            return None

        # Optional: check resolution
        if image.shape[:2] != (self.frame_height, self.frame_width):
            print('image shape not matching')
            print(f"Unexpected image shape: {image.shape[:2]}, expected: {(self.frame_height, self.frame_width)}")
            return None

        return image  # Already in BGR format
    

    def process_compressed__grayscale_frame(self, jpeg_data):

        # Convert bytes to a 1D uint8 numpy array
        buffer = np.frombuffer(jpeg_data, dtype=np.uint8)
        
        # Decode the JPEG image directly as grayscale
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        
        if image is None:
            print("Failed to decode JPEG image")
            return None
        
        # Optional: check shape if needed
        if image.shape != (self.frame_height, self.frame_width):
            print(f"Unexpected image size: {image.shape}, expected: {(self.frame_height, self.frame_width)}")
            return None
        
        return image
    
    
    def compute_fps(self, last_time, frame_count, fps):

        frame_count += 1
        current_time = time.time()
        elapsed = current_time - last_time
        
        # Update FPS every second
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            last_time = current_time

        return fps, frame_count, last_time
    
    
    def get_image(self, client_socket):

        frame = self.read_frame(client_socket)

        if frame is None:
            return None
        
        image = self.process_compressed_rgb565_frame(frame)
        
        return image