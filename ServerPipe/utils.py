import math
import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pynput import keyboard

class Utils:

    def __init__(self):
        self.pressed_keys = set()
        self.listener = self.start_listener()


    def rotation_matrix(self, pitch, roll):
        # Assuming yaw = 0
        pitch = math.radians(pitch)
        roll = math.radians(roll)

        Rx = np.array([
            [1, 0, 0],
            [0, math.cos(pitch), -math.sin(pitch)],
            [0, math.sin(pitch), math.cos(pitch)]
        ])

        Rz = np.array([
            [math.cos(roll), 0, -math.sin(roll)],
            [0, 1, 0],
            [math.sin(roll), 0, math.cos(roll)],
        ])
        
        return Rz @ Rx
    

    def integrate(self, prev, curr, dt):
        return (prev + curr)/2 * dt
    

    def integrate_btw_frames(self, accel_btw_frames, v0, d):
        v1 = v0.copy()
        d = np.zeros(3)
        dt_tot = 0.0

        for i in range(len(accel_btw_frames) - 1):

            a0 = accel_btw_frames[i]['a']
            a1 = accel_btw_frames[i+1]['a']
            dt = accel_btw_frames[i+1]['dt']

            # Integrate acceleration to get velocity
            v1 += (a0 + a1) * 0.5 * dt
            
            dt_tot += dt

        # Integrate velocity to get displacement
        delta_d = (v0 + v1) * 0.5 * dt_tot

        d += delta_d

        return v1, d
    

    def save_image(self, image, frame_count, Folder):

        if not os.path.exists(Folder):
            print(f"Error: Folder '{Folder}' does not exist!")
            return
        elif not os.access(Folder, os.W_OK):
            print(f"Error: No write permission for folder '{Folder}'!")
            return

        # Save the image frame
        folder_name = os.path.basename(os.path.normpath(Folder))
        frame_filename = os.path.join(Folder, f"{folder_name}_frame_{frame_count}.jpg")
        success = cv2.imwrite(frame_filename, image)

        if not success:
            print("Failed to save image. Check the image format and path.")


    def save_tof_values(self, tof_value, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"tof.npy")
        # Convert scalar to array before concatenation
        tof_value = np.array([tof_value])
        if os.path.exists(save_path):
            existing = np.load(save_path)
            combined = np.concatenate([existing, tof_value])
            np.save(save_path, combined)
        else:
            np.save(save_path, tof_value)

    
    def save_imu_values(self, imu_values, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"imu.npy")
        
        gyro = imu_values['gyro']
        accel = imu_values['accel']

        # Convert gyro and accel to numpy arrays
        gyro = np.array(gyro, dtype=np.float32)
        accel = np.array(accel, dtype=np.float32)
        imu_data = np.concatenate((gyro, accel))
        if os.path.exists(save_path):
            existing = np.load(save_path)
            combined = np.concatenate([existing, imu_data])
            np.save(save_path, combined)
        else:
            np.save(save_path, imu_data)

    
    def save_object_detection_data(self, object_detected, bounding_box_coords, center, frame_timestamp,  frame_count, folder_name, save_dir):
        
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'object_detection.npy')

        image_name = f"{folder_name[7:]}_frame_{frame_count}.jpg"

        # Create the data entry as a dictionary
        if object_detected:
            data_entry = {
                'image_name': image_name,
                'timestamp': float(frame_timestamp),
                'detected': True,
                'bbox': {
                    'x1': float(bounding_box_coords[0][0]),
                    'y1': float(bounding_box_coords[0][1]),
                    'x2': float(bounding_box_coords[1][0]),
                    'y2': float(bounding_box_coords[1][1]),
                },
                'center': {
                    'x': float(center[0]),
                    'y': float(center[1]),
                }
            }
        else:
            data_entry = {
                'image_name': image_name,
                'timestamp': float(frame_timestamp),
                'detected': False,
                'bbox': None,
                'center': None
            }

        # Load existing data if available
        if os.path.exists(save_path):
            existing = list(np.load(save_path, allow_pickle=True))
            existing.append(data_entry)
            np.save(save_path, existing)
        else:
            np.save(save_path, [data_entry])


    def load_junctions(self, filepath):
        with open(filepath, 'r') as f:
            junctions = json.load(f)
        return junctions


    def initialize_orientation_plot(self):
        # Set up matplotlib for 3D orientation plot
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.set_zlabel("Y")
        ax.set_title("IMU Orientation (3D)")
        ax.view_init(elev=30, azim=135)

        return fig, ax

    def plot_orientation(self, ax, R):

        origin = np.array([[0, 0, 0]]).T
        x_axis = R @ np.array([[1, 0, 0]]).T
        y_axis = R @ np.array([[0, 1, 0]]).T
        z_axis = R @ np.array([[0, 0, 1]]).T

        ax.cla()
        ax.quiver(*origin.flatten(), *x_axis.flatten(), color='r', label='X')
        ax.quiver(*origin.flatten(), *y_axis.flatten(), color='b', label='Z')
        ax.quiver(*origin.flatten(), *z_axis.flatten(), color='g', label='Y')
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([1, -1])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("IMU Orientation (3D)")
        ax.legend()
        ax.view_init(elev=30, azim=135)

        plt.gcf().canvas.draw_idle()
        plt.gcf().canvas.start_event_loop(0.1)

    # Keyboard 
    def start_listener(self):
        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()
        return listener

    def on_press(self, key):
        try:
            self.pressed_keys.add(key.char)
        except AttributeError:
            pass

    def on_release(self, key):
        try:
            self.pressed_keys.discard(key.char)
            if key.char == 'q':
                return False  # Stop listener
        except AttributeError:
            pass

    def is_pressed(self, key):
        return key in self.pressed_keys
