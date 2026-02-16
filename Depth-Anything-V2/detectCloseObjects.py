import cv2
import torch
import time
import numpy as np
import threading
from collections import deque
import socket

from depth_anything_v2.dpt import DepthAnythingV2
from vision import *
from image_processing import *

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vitb' # or 'vits', 'vitb', 'vitg'

model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
model = model.to(DEVICE).eval()

# Wifi
HOST = '172.20.10.13'
PORT = 12345
MAX_ARRAY_LENGTH = 500

image_lock = threading.Lock()
image_queue = deque(maxlen=1)


def compute_fps(start_time, last_time, frame_count, fps):

    frame_count += 1
    current_time = time.time()
    elapsed = current_time - last_time

    # Update FPS every second
    if elapsed >= 1.0:
        fps = frame_count / elapsed
        frame_count = 0
        last_time = current_time

    return fps, frame_count, last_time

def load_camera_matrix():
        with open('calib1.txt', 'r') as f:
            lines = f.readlines()
        
        P = np.zeros((3, 4))
        for i in range(3):
            P[i] = list(map(float, lines[i].strip().split()))

        K = P[:, :3]
        
        return K, P


def handle_camera_client(client_socket):

    total_frame_count = 0
    # FPS calculation variables
    frame_count = 0
    last_time = time.time()
    fps = 0

    while True:
        try:
            start_time = time.time()

            image = get_image(client_socket)
            # image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

            toF_distance = read_ToF_value(client_socket) # mm

            # depth map
            depth = model.infer_image(image) # HxW raw depth map in numpy
            depth = cv2.GaussianBlur(depth, (5,5), 0)
            depth_normalized, depth_colored = normalize_and_colormap(depth)

            # Depth gradient
            depth_gradient = get_depth_gradient(depth)
            depth_gradient_normalized, depth_gradient_colored = normalize_and_colormap(depth_gradient)

            # Compute grad - distance score
            score = dist_grad_score(depth, depth_gradient, alpha=0.3, beta=0.7)
            score_normalized, score_colored = normalize_and_colormap(score)
            filtered_high_score = large_score(score, threshold=0.5)

            # Keep only pixels in the foreground
            depth_gradient_front = remove_far_elements(depth_gradient, depth, std_fact = 0.2)

            # Keep only pixels with large gradient
            depth_gradient_front = large_gradient(depth_gradient_front, threshold = 70, ratio = 0.6)

            # Obstacle bounding box
            obstacle, image_with_obstacle, bounding_box_coords, obj_center = detect_obstacle(image, depth, filtered_high_score, tol=0.2)

            if obstacle and toF_distance:
                # Scaled depth map based on toF value
                scaled_depth_map = scale_depth_map(depth_normalized, toF_distance, obj_center)

                # Obstacle distance
                if scaled_depth_map is None or obj_center is None:
                    pass
                else:
                    obj_distance = scaled_depth_map[obj_center]
                    # distance on image
                    dist_text = f"DIST: {toF_distance:.2f} mm"
                    image_with_obstacle = cv2.putText(image_with_obstacle, dist_text, (10,60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                                (0, 255, 0), 2)
                    
                # Obstacle height
                K, _ = load_camera_matrix()
                height = estimate_obj_height(bounding_box_coords, obj_distance, K, cam_height = 50)
                dimension_text = f"Height: {height:.2f} mm"
                image_with_obstacle = cv2.putText(image_with_obstacle, dimension_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)

            # Frame rate
            fps, frame_count, last_time = compute_fps(start_time, last_time, frame_count, fps)

            # Handle case when no obstacle detected
            if obstacle:
                displayed_image = image_with_obstacle
            else:
                displayed_image = image
            
            try:
                displayed_image = add_fps_text(displayed_image, fps)
                with image_lock:
                    image_infos = {
                        "image": displayed_image,
                        "depth_image": depth_colored,
                        "depth_gradient": depth_gradient_front,
                        "score": score_colored,
                        "high_score": filtered_high_score,
                        "timestamp": time.time(),
                        "frame_count": total_frame_count,
                        "ToF_value": toF_distance
                    }
                    image_queue.append(image_infos) 
                    
            except Exception as e:
                print(f"Display error: {str(e)}")
                with image_lock:
                    image_infos = {
                        "image": displayed_image,
                        "depth_image": depth,
                        "depth_gradient": depth_gradient_front,
                        "score": score_colored,
                        "high_score": filtered_high_score,
                        "timestamp": time.time(),
                        "frame_count": total_frame_count,
                        "ToF_value": toF_distance
                    }
                    image_queue.append(image_infos)  # Fallback to original image
            
            total_frame_count += 1

        except (ConnectionResetError, OSError):
            break

    client_socket.close()


# Function to start the server and accept clients
def start_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, PORT))
    server.listen(5)
    print(f"[*] Listening on {HOST}:{PORT}")
    while True:
        client_socket, addr = server.accept()
        print(f"[*] Accepted connection from {addr}")
        client_handler = threading.Thread(target=handle_camera_client, args=(client_socket,))
        client_handler.start()


def main():

     # Start server thread
    server_thread = threading.Thread(target=start_server)
    server_thread.daemon = True
    server_thread.start()

    while True:
        # Check for images to display
        with image_lock:
            if image_queue:
                image_infos = image_queue.popleft()
                image = image_infos["image"]
                depth_image = image_infos["depth_image"]
                depth_gradient = image_infos["depth_gradient"]
                score = image_infos["score"]
                high_score = image_infos["high_score"]


                cv2.imshow("Image", image)
                cv2.imshow("Depth", depth_image)
                cv2.imshow("Depth Gradient", depth_gradient)
                cv2.imshow("Score", score)
                cv2.imshow("High score", high_score)

                cv2.moveWindow("Nicla Vision - RGB", 0, 0)
                cv2.moveWindow("Nicla Vision - Depth", 640, 0)
                cv2.moveWindow("Nicla Vision - Depth Gradient", 1280, 0)
        
        cv2.waitKey(1)


if __name__ == "__main__":
    main()