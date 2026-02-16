import numpy as np
import math
from scipy.spatial.transform import Rotation as R

G = 9.1

class ImuProcessor:

    def __init__(self):
        self.accel_offset = [0.01830895, 0.17269031, 0.23701544]
        self.gyro_offset = [-0.42391914, -0.10809209, 0.15127149]

        self.gyro_angles = [0, 0, 0]
        self.orientation = [0, 0, 0]


    def correct_offset(self, imu_raw):
        
        imu_raw["accel"] = np.add(imu_raw["accel"], self.accel_offset)
        imu_raw["gyro"] = np.add(imu_raw["gyro"], self.gyro_offset)

        return imu_raw
    

    def remap_to_camera_frame(self, imu_values):
        ax, ay, az = imu_values['accel']
        gx, gy, gz = imu_values['gyro']

        accel_camera_frame = [ay, ax, az]  # Accel Y = right → camera X = right, accel X = down → camera Y = down, Z (forward) is aligned
        gyro_camera_frame = [-gy, -gx, -gz] # Gyro Y = left → camera X = right, gyro X = up → camera Y = down, Z (forward) is aligned

        return np.array(accel_camera_frame), np.array(gyro_camera_frame)
    
    
    def get_world_accel(self, accel_raw, imu_orientation):

        # Subtract gravity vector from body-frame measurement
        gravity_body = imu_orientation.apply(np.array([0, 9.81, 0]))  # what gravity looks like in IMU frame
        accel_body_corrected = accel_raw - gravity_body
        # accel_body_corrected = accel_raw - [0, 9.81, 0]

        # Rotate to world frame
        accel_world = imu_orientation.inv().apply(accel_body_corrected)
        return accel_world


    def compute_orientation(self, imu_values, dt):

        # Accelerometer
        ax = imu_values['accel'][0]
        ay = imu_values['accel'][1]
        az = imu_values['accel'][2]

        pitch_a = (180/math.pi) * np.arctan2(az, ay)
        roll_a = (180/math.pi) * np.arctan2(ax, ay)

        # Gyro
        self.gyro_angles += imu_values['gyro'] * dt

        # Complementary filter (further replace by Kalman)
        pitch = 0.98 * pitch_a + 0.02 * self.gyro_angles[0]
        roll = 0.98 * roll_a + 0.02 * self.gyro_angles[1]
        yaw = self.gyro_angles[2]

        return pitch, roll, yaw

    def instant_rotation(self, gyro_values_btw_frames):
        """Integrate gyroscope values over a window and compute rotation"""
        if len(gyro_values_btw_frames) < 2:
            return np.eye(3)
        
        # Start with identity rotation
        total_rotation = R.identity()
        
        for i in range(len(gyro_values_btw_frames)-1):
            g0 = np.array(gyro_values_btw_frames[i]['gyro'])
            g1 = np.array(gyro_values_btw_frames[i+1]['gyro'])
            dt = gyro_values_btw_frames[i+1]['dt']
            
            # Trapezoidal integration for angular velocity
            avg_angular_velocity = (g1 + g0) * 0.5
            
            # Integrate to get angular displacement (in radians)
            angular_displacement = avg_angular_velocity * dt
            
            # Create rotation from angular displacement
            # incremental_rotation = R.from_euler('xyz', angular_displacement, degrees=True)
            incremental_rotation = R.from_rotvec(np.deg2rad(angular_displacement))
            
            # Compose rotations (multiplication order matters!)
            total_rotation = total_rotation * incremental_rotation
        
        return total_rotation.as_matrix()