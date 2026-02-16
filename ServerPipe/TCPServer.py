import socket
import threading
import time
import os
from collections import deque
import queue
import numpy as np
import cv2
import traceback

from nicla_vision_stream import *
from image_processing import *
from object_detector import *
from pose_estimator import *
from imu_processing import ImuProcessor
from utils import Utils
from vo import VisualOdometry
from turn_classifier import TurnClassifier
from robot_controller import RobotController
from map import Mapper

HOST = '192.168.4.2'
CAMERA_PORT = 12345
IMU_PORT = 12346

SAVE_IMAGE = False
SAVE_TOF_VALUES = False
SAVE_IMU_VALUES = False
SAVE_JUNCTION_VALUES = False
SAVE_OBJ_DETECT_DATA = False

SHOW_STATS = False

AUTOMATIC_ROBOT_CONTROL = False

MAPPING = False

# Frame rate control
TARGET_CAMERA_FPS = 10.0  # Target
TARGET_IMU_HZ = 30.0    # Target
MAX_FRAME_BUFFER = 1    # Maximum frames to buffer (drop older ones)
MAX_IMU_BUFFER = 10      # Maximum IMU readings to buffer

# Thread-safe queues for decoupled processing
raw_image_queue = queue.Queue(maxsize=MAX_FRAME_BUFFER)
processed_image_queue = queue.Queue(maxsize=5)
raw_imu_queue = queue.Queue(maxsize=MAX_IMU_BUFFER)

# Display queue
image_queue = deque(maxlen=10)
image_lock = threading.Lock()
total_frame_count = 0

# Folder to store images
newFolder = ""

# Image processing parameters
FRAME_HEIGHT = 320
FRAME_WIDTH = 240
BYTES_PER_PIXEL = 3

# Initialize classes
utils = Utils()
niclaVisionStreamer = NiclaVisionStreamer(FRAME_HEIGHT, FRAME_WIDTH, BYTES_PER_PIXEL)
imageProcessor = ImageProcessor(score_threshold=0.3, depth_model='vitb')
imuProcessor = ImuProcessor()
objectDetector = ObjectDetector()
junctionDetector = JunctionDetector('calib1.txt', 10)
pose_estimator = PoseEstimator()
vo = VisualOdometry(calib_file='calib1.txt')
turn_classifier = TurnClassifier(model_path='best_turn_classifier_45_90_improved.pth', class_names=['left45', 'left90', 'right45', 'right90', 'straight'])
robot_controller = RobotController()
mapper = Mapper(pipe_seg_length=25.0, standard_turn_angles=[-90, -45, 0, 45, 90])

# IMU data
orientation = {"pitch": 0.0, "roll": 0.0, "yaw": 0.0}
lastest_accel = None
lastest_gyro = None
accel_btw_frames = []
gyro_btw_frames = []

# ToF data
lastest_tof = None

# Poses
cam_poses = []
absolute_poses = []

rotation_enabled = False

# Locks
imu_data_lock = threading.Lock()
tof_data_lock = threading.Lock()
poses_lock = threading.Lock()
ekf_lock = threading.Lock()
orietnation_estimator_lock = threading.Lock()
controller_lock = threading.Lock()

# Performance monitoring
camera_stats = {
    'received_frames': 0,
    'processed_frames': 0,
    'dropped_frames': 0,
    'last_fps_time': time.time(),
    'fps': 0.0
}

imu_stats = {
    'received_packets': 0,
    'processed_packets': 0,
    'dropped_packets': 0,
    'last_hz_time': time.time(),
    'hz': 0.0
}

stats_lock = threading.Lock()

# Ground truth junction positions
junctions = utils.load_junctions('data/junctions.json')


def handle_camera_reception(client_socket):
    """
    Thread dedicated to receiving camera frames at constant rate
    """
    global camera_stats
    
    frame_interval = 1.0 / TARGET_CAMERA_FPS
    next_frame_time = time.time()
    
    while True:
        try:
            current_time = time.time()
            
            # Receive image
            image = niclaVisionStreamer.get_image(client_socket)

            if image is None:
                continue

            # Uncomment this line if the camera is upside down
            # image = cv2.rotate(image, cv2.ROTATE_180)

            if SAVE_IMAGE:
                utils.save_image(image, total_frame_count, newFolder)
                
            with stats_lock:
                camera_stats['received_frames'] += 1
            
            # Create frame data with timestamp
            frame_data = {
                'image': image,
                'timestamp': current_time,
                'frame_id': camera_stats['received_frames'],
            }
            
            # Try to add to queue, drop oldest if full
            try:
                raw_image_queue.put_nowait(frame_data)
            except queue.Full:
                try:
                    # Remove oldest frame and add new one
                    raw_image_queue.get_nowait()
                    raw_image_queue.put_nowait(frame_data)
                    with stats_lock:
                        camera_stats['dropped_frames'] += 1
                except queue.Empty:
                    pass
            
            # Frame rate control - sleep until next frame time
            next_frame_time += frame_interval
            sleep_time = next_frame_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # If we're behind, adjust next frame time
                next_frame_time = time.time()
                
        except (ConnectionResetError, OSError):
            break
    
    client_socket.close()


def handle_camera_processing():
    """
    Thread dedicated to processing camera frames
    """
    global total_frame_count, pose, lastest_accel, lastest_gyro
    global cam_poses, orientation
    global accel_btw_frames, gyro_btw_frames, camera_stats
    global rotation_enabled
    
    last_time = time.time()

    straight_junction_added = False
    
    while True:
        try:
            # Get latest frame (blocking)
            frame_data = raw_image_queue.get(timeout=1.0)
            image = frame_data['image']
            frame_timestamp = frame_data['timestamp']
            
            with stats_lock:
                camera_stats['processed_frames'] += 1
            
            dt = time.time() - last_time
            last_time = time.time()

            # Predict next turn direction
            turn_direction, confidence = turn_classifier.predict(image)

            # Map
            if MAPPING:
                if robot_controller.next_direction == 'straight' and not straight_junction_added:
                    if junctionDetector.detection_state == 1:
                        print('here')
                        mapper.add_junction(robot_controller.next_direction, junctionDetector.last_junction_id+1) # Add junction with 0 angle if no turn detected
                        straight_junction_added = True
                else:
                    if robot_controller.turn_started():
                        print('TURN START')
                        mapper.add_junction(robot_controller.next_direction, junctionDetector.last_junction_id+1)
                        robot_controller.set_turn_started(False)
                    if robot_controller.turn_ended():
                        print('TURN END')
                        robot_controller.set_turn_ended(False)
                        print(f'add junction with turn: {robot_controller.next_direction}')

            if junctionDetector.detection_state == 0:
                straight_junction_added = False

            # Robot control
            if AUTOMATIC_ROBOT_CONTROL:
                with controller_lock:
                    robot_controller.update_direction(turn_direction, confidence)
            
            # Process image
            image_with_circles, circles = junctionDetector.detect_circles(image)
            if circles is not None:
                larger_circle = max(circles, key=lambda c: c['radius'])
                distance = junctionDetector.compute_dist_to_junc(junction_rad=3.75, circle=larger_circle)

            # Junction detection
            junctionDetector.windowed_junction_detection()

            if SAVE_JUNCTION_VALUES and junctionDetector.detection_state == 1:
                # Save distance to npy file
                distances_file = os.path.join('data', 'junctions', 'distances.npy')
                # Load existing distances if file exists, else start new array
                if os.path.exists(distances_file):
                    distances = np.load(distances_file).tolist()
                else:
                    distances = []
                distances.append(distance)
                np.save(distances_file, np.array(distances))

            # Get IMU data for this time period
            with imu_data_lock:
                R_imu = imuProcessor.instant_rotation(gyro_btw_frames.copy())
                gyro_btw_frames.clear()
                
            yaw = np.arctan2(R_imu[0, 2], R_imu[2, 2])

            # Build yaw-only rotation matrix (Y-down frame)
            R_yaw = np.array([
                [ np.cos(yaw), 0, np.sin(yaw)],
                [ 0,           1, 0          ],
                [-np.sin(yaw), 0, np.cos(yaw)]
            ])

            # Integrate IMU data between frames
            with imu_data_lock:
                new_v, new_d = utils.integrate_btw_frames(accel_btw_frames.copy(), np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]))
                accel_btw_frames.clear()

            with poses_lock:
                if junctionDetector.detection_state == 1 and MAPPING:
                        try:
                            pose = pose_estimator.update_pose_with_junction(mapper.junctions, junctionDetector.last_junction_id, distance)
                            cam_poses.append(pose[:3, 3].copy())
                        except:
                            print(f'Pose estimation failed: Junction ID {junctionDetector.last_junction_id} is out of bounds of the known junctions.')
                        

            # Object detection
            depth = imageProcessor.generate_depth(image)
            depth_colored, depth_normalized, depth_high_laplacian, depth_canny, edges = imageProcessor.depth_processing(depth)
            if lastest_tof < 250:
                object_detected, image_with_obstacle, bounding_box_coords, center = objectDetector.detect_obstacle(image, edges)
            else:
                image_with_obstacle = image.copy()

            # Save object detection data
            if SAVE_OBJ_DETECT_DATA:
                utils.save_object_detection_data(object_detected, bounding_box_coords, center, frame_timestamp, total_frame_count, newFolder, save_dir='data/object_detection')

            # Prepare processed result
            processed_result = {
                'image': image.copy(),
                'junctions': image_with_circles,
                'object_detection': image_with_obstacle,
                'depth': depth_colored,
                'edges': edges,
                'timestamp': frame_timestamp,
                'frame_count': total_frame_count,
                'processing_latency': time.time() - frame_timestamp
            }
            
            # Add to processed queue (non-blocking, drop if full)
            try:
                processed_image_queue.put_nowait(processed_result)
            except queue.Full:
                try:
                    processed_image_queue.get_nowait()
                    processed_image_queue.put_nowait(processed_result)
                except queue.Empty:
                    pass
            
            total_frame_count += 1
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Processing error: {e}")
            traceback.print_exc()


def handle_imu_tof_reception(client_socket):
    """
    Thread dedicated to receiving IMU data at constant rate
    """
    global imu_stats, lastest_tof
    
    while True:
        try:
            current_time = time.time()
            
            # Receive IMU data
            imu_packet = niclaVisionStreamer.read_imu_data(client_socket)
            if imu_packet is None:
                continue
            imu_raw, timestamp = imu_packet

            if SAVE_IMU_VALUES:
                utils.save_imu_values(imu_raw, save_dir='data/imu')

            tof_value = niclaVisionStreamer.read_ToF_value(client_socket)
            if tof_value is None: 
                continue

            if SAVE_TOF_VALUES:
                utils.save_tof_values(tof_value, save_dir='data/tof')
            with tof_data_lock:
                lastest_tof = tof_value
            
            with stats_lock:
                imu_stats['received_packets'] += 1
            
            # Create IMU data packet with timestamp
            imu_tof_packet = {
                'data': imu_raw,
                'timestamp': timestamp/1_000_000.0,
                'packet_id': imu_stats['received_packets'],
                'tof_value': tof_value
            }
            
            # Try to add to queue, drop oldest if full
            try:
                raw_imu_queue.put_nowait(imu_tof_packet)
            except queue.Full:
                try:
                    raw_imu_queue.get_nowait()
                    raw_imu_queue.put_nowait(imu_tof_packet)
                    with stats_lock:
                        imu_stats['dropped_packets'] += 1
                except queue.Empty:
                    pass
                    
        except Exception as e:
            print(f"IMU reception error: {e}")
            traceback.print_exc()
            break
    
    client_socket.close()


def handle_imu_tof_processing():
    """
    Thread dedicated to processing IMU data
    """
    global orientation, lastest_accel, lastest_gyro
    global accel_btw_frames, gyro_btw_frames, imu_stats
    global rotation_enabled

    last_packet_timestamp = None
    
    while True:
        try:
            # Get IMU packet (blocking with timeout)
            imu_tof_packet = raw_imu_queue.get(timeout=1.0)
            imu_raw = imu_tof_packet['data']
            packet_timestamp = imu_tof_packet['timestamp']
            tof_value = imu_tof_packet['tof_value']
            
            with stats_lock:
                imu_stats['processed_packets'] += 1

            if last_packet_timestamp is not None:
                dt = packet_timestamp - last_packet_timestamp
            else:
                dt = 0.0 

            # Update last timestamp
            last_packet_timestamp = packet_timestamp
            
            # Process IMU data
            imu_values = imuProcessor.correct_offset(imu_raw)
            imu_values['accel'], imu_values['gyro'] = imuProcessor.remap_to_camera_frame(imu_values)
            pitch, roll, yaw = imuProcessor.compute_orientation(imu_values, dt)

            if AUTOMATIC_ROBOT_CONTROL:
                with controller_lock:
                    robot_controller.update_command(tof_value, tof_low_threshold=30, tof_high_threshold=200)

            with imu_data_lock:
                orientation["pitch"] = pitch
                orientation["roll"] = roll

                lastest_accel = np.array(imu_values['accel'])
                lastest_gyro = np.array(imu_values['gyro'])

                orient = R.identity()
                accel = imuProcessor.get_world_accel(lastest_accel, orient)
                accel_btw_frames.append({'a': accel, 'dt': dt})
                gyro_btw_frames.append({'gyro': imu_values['gyro'], 'dt': dt})
                
        except queue.Empty:
            continue
        except Exception as e:
            print(f"IMU processing error: {e}")
            print(traceback.format_exc())


def handle_display():
    """
    Thread dedicated to display and UI updates
    """
    global camera_stats, imu_stats
    global cam_poses
    global AUTOMATIC_ROBOT_CONTROL
    
    # Initialize plots
    if MAPPING:
        map_fig, map_axes = mapper.initialize_map_plot()
    
    while True:

        # Start/stop robot control
        if utils.is_pressed('s'):
            AUTOMATIC_ROBOT_CONTROL = not AUTOMATIC_ROBOT_CONTROL
            if AUTOMATIC_ROBOT_CONTROL:
                robot_controller.start()
            else:
                robot_controller.stop()

        try:
            # Get latest processed frame
            processed_result = processed_image_queue.get(timeout=0.1)
            
            # Calculate and display FPS
            current_time = time.time()
            with stats_lock:
                # Update camera FPS
                if current_time - camera_stats['last_fps_time'] >= 1.0:
                    camera_stats['fps'] = camera_stats['processed_frames'] / (current_time - camera_stats['last_fps_time'])
                    camera_stats['processed_frames'] = 0
                    camera_stats['last_fps_time'] = current_time
                
                # Update IMU Hz
                if current_time - imu_stats['last_hz_time'] >= 1.0:
                    imu_stats['hz'] = imu_stats['processed_packets'] / (current_time - imu_stats['last_hz_time'])
                    imu_stats['processed_packets'] = 0
                    imu_stats['last_hz_time'] = current_time
            
            # Add FPS and latency info to image
            image_to_display = processed_result['image'].copy()
            latency_ms = processed_result['processing_latency'] * 1000
            
            if SHOW_STATS:
                with stats_lock:
                    imageProcessor.add_stats_on_image(camera_stats, imu_stats, latency_ms, image_to_display)
            
            # Display images
            cv2.imshow("Image", image_to_display)
            cv2.imshow("Object Detection", processed_result['object_detection'])
            #cv2.imshow("Depth", processed_result['depth'])
            #cv2.imshow("Edge Detection", processed_result['edges'])
            cv2.imshow("Junction Detection", processed_result['junctions'])
            
            # Position windows
            cv2.moveWindow("Image", 0, 0)
            cv2.moveWindow("Object Detection", 320, 0)
            #cv2.moveWindow("Depth", 0, 240)
            #cv2.moveWindow("Edge Detection", 320, 240)
            cv2.moveWindow("Junction Detection", 0, 240)
            
            # Update pose plot
            if MAPPING:
                with poses_lock:
                    mapper.update_map_plot(map_axes, cam_poses)

            # If r key is pressed, reset the pose and the pose plot
            if utils.is_pressed('r'):
                with poses_lock:
                    pose_estimator.reset_pose()
                    cam_poses.clear()
                    cam_poses.append(pose_estimator.pose[:3, 3].copy())
                    junctionDetector.reset_detection()

            if utils.is_pressed('m'):
                # Save map plot
                map_fig.savefig(f"map_{int(time.time())}.png")
            
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Display error: {e}")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def start_camera_server():
    camera_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    camera_server.bind((HOST, CAMERA_PORT))
    camera_server.listen(5)
    print(f"[*] Listening for camera on {HOST}:{CAMERA_PORT}")
    
    while True:
        client_socket, addr = camera_server.accept()
        print(f"[*] Accepted camera connection from {addr}")
        
        # Start reception thread
        reception_thread = threading.Thread(
            target=handle_camera_reception, 
            args=(client_socket,),
            daemon=True
        )
        reception_thread.start()


def start_imu_server():
    imu_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    imu_server.bind((HOST, IMU_PORT))
    imu_server.listen(5)
    print(f"[*] Listening for IMU on {HOST}:{IMU_PORT}")
    
    while True:
        client_socket, addr = imu_server.accept()
        print(f"[*] Accepted IMU connection from {addr}")
        
        # Start reception thread
        reception_thread = threading.Thread(
            target=handle_imu_tof_reception, 
            args=(client_socket,),
            daemon=True
        )
        reception_thread.start()


def main():
    global newFolder
    
    if SAVE_IMAGE:
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        newFolder = "Images/run_" + timestamp
        os.makedirs(newFolder, exist_ok=True)
        os.chmod(newFolder, 0o777)

    # Start network servers
    camera_thread = threading.Thread(target=start_camera_server, daemon=True)
    imu_thread = threading.Thread(target=start_imu_server, daemon=True)
    camera_thread.start()
    imu_thread.start()

    # Start processing threads
    camera_processing_thread = threading.Thread(target=handle_camera_processing, daemon=True)
    imu_processing_thread = threading.Thread(target=handle_imu_tof_processing, daemon=True)
    camera_processing_thread.start()
    imu_processing_thread.start()

    # Start robot controller
    robot_controller.start_communcation()

    # Run display in main thread
    handle_display()

    # Stop robot controller on exit
    robot_controller.stop_communication()


if __name__ == "__main__":
    main()