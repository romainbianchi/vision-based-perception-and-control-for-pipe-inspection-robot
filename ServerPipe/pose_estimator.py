import math
import numpy as np
from scipy.spatial.transform import Rotation as R

class PoseEstimator:

    def __init__(self):
        self.last_time = None
        self.total_angle = 0.0
        self.pose = np.eye(4, dtype=np.float64)
        self.pose[:3, 3] = np.array([0.0, 0.0, 0.0])  # Initial position at the origin

    
    def fuse_rotation(self, R_vo, R_imu, vo_available, alpha=0.1):

        if not vo_available:
            return R_imu
        
        # High-pass IMU (preserves fast motion)
        # Low-pass VO (provides drift correction)
        r_imu = R.from_matrix(R_imu).as_rotvec()
        r_vo = R.from_matrix(R_vo).as_rotvec()
        
        # IMU contributes high-frequency, VO contributes low-frequency
        r_fused = alpha * r_imu + (1-alpha) * r_vo
        
        return R.from_rotvec(r_fused).as_matrix()
    
    def compute_robot_pos(self, junctions, junction_id, distance):
        actual_junction = junctions[junction_id]
        pos = actual_junction['pos']

        # accumulate all previous junction angles to get actual angle
        junctions_angles = [junction['angle'] for junction in junctions]
        junctions_angles = junctions_angles[:junction_id+1]
        self.total_angle = np.sum(junctions_angles)
        total_angle_rad = np.deg2rad(self.total_angle)

        # Compute robot position relatively to the junction
        x = pos[0] + np.cos(math.pi/2 + total_angle_rad) * distance
        z = pos[1] - np.sin(math.pi/2 + total_angle_rad) * distance

        return x, 0.0, z
    
    def update_pose(self, translation, rotation):

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = rotation
        T[:3, 3] = translation
        
        self.pose = np.matmul(self.pose, np.linalg.inv(T))

        return self.pose
    

    def update_pose_with_junction(self, junctions, junction_id, distance):
        junction_id = junction_id

        if junction_id <= len(junctions):
            robot_x, robot_y, robot_z = self.compute_robot_pos(junctions, junction_id, distance)
            angle = self.total_angle
            angle_rad = np.deg2rad(angle)
            # only consider yaw angle
            orient = np.array([
                [np.cos(angle_rad), 0, np.sin(angle_rad)],
                [ 0,           1, 0          ],
                [-np.sin(angle_rad), 0, np.cos(angle_rad)]
            ])
            self.pose[:3, 3] = np.array([robot_x, 0, robot_z])
            self.pose[:3, :3] = orient
            
            return self.pose
        
        else:
            print(f"Junction ID {junction_id} is out of bounds for the junctions list.")
            return self.pose
        
    def reset_pose(self):
        self.pose = np.eye(4, dtype=np.float64)
        self.pose[:3, 3] = np.array([0.0, 0.0, 0.0])
        self.total_angle = 0.0

        return self.pose

