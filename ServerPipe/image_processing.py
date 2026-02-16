import sys
import os
import cv2
import numpy as np
import torch

# Add path to Depth Anything-V2
sys.path.append('../Depth-Anything-V2')
from depth_anything_v2.dpt import DepthAnythingV2


class ImageProcessor:

    def __init__(self, score_threshold, depth_model):

        self.score_threshold = score_threshold

        # Depth model
        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        self.encoder = depth_model
        self.model = DepthAnythingV2(**self.model_configs[self.encoder])
        self.model.load_state_dict(torch.load(f'../Depth-Anything-V2/checkpoints/depth_anything_v2_{self.encoder}.pth', map_location='cpu'))
        self.model = self.model.to(self.device).eval()

    
    def generate_depth(self, image):
        return self.model.infer_image(image)
        

    def normalize_and_colormap(self, image):
        image_normalized_0_255 = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        image_normalized = cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX)
        image_uint8 = image_normalized_0_255.astype(np.uint8)
        image_colored = cv2.applyColorMap(image_uint8, cv2.COLORMAP_JET)

        return image_normalized, image_colored
    
    
    def get_depth_gradient(self, depth):
        # Compute the gradient of the depth map
        gradient_x = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=5)
        gradient_y = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=5)

        # Compute the magnitude of the gradient
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

        return gradient_magnitude
    
    
    def get_depth_laplacian(self, depth):
        return np.abs(cv2.Laplacian(depth, cv2.CV_64F, ksize=5))
    
    
    def get_high_laplacian(self, depth_laplacian):

        max_lap = np.max(depth_laplacian)

        condition = (depth_laplacian/max_lap)**2 > self.score_threshold
        high_laplacian = np.zeros_like(depth_laplacian, dtype=np.uint8)
        high_laplacian[condition] = 255

        return high_laplacian

        
    def get_depth_canny(self, depth, sigma=0.33):
        depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_uint8 = depth_norm.astype(np.uint8)

        median = np.median(depth_uint8)
        low_thresh = int(max(0, (1.0 - sigma) * median))
        up_thresh = int(min(255, (1.0 + sigma) * median))

        edges = cv2.Canny(depth_uint8, threshold1=low_thresh, threshold2=up_thresh)

        return edges
    
    def filter_edges(self, depth_canny, depth_gradient, percentile=99):
        # keep only edges with high gradient
        threshold = np.percentile(depth_gradient, percentile)
        filtered_edges = np.zeros_like(depth_canny, dtype=np.uint8)
        condition = depth_gradient > threshold
        filtered_edges[condition] = depth_canny[condition]

        return filtered_edges


    def depth_processing(self, depth):
        depth_smoothed = cv2.GaussianBlur(depth, (5,5), 1.5)
        depth_normalized, depth_colored = self.normalize_and_colormap(depth)
        depth_laplacian = self.get_depth_laplacian(depth_smoothed)
        depth_high_laplacian = self.get_high_laplacian(depth_laplacian)
        depth_gradient = self.get_depth_gradient(depth_smoothed)
        depth_canny = self.get_depth_canny(depth_smoothed)
        edges = self.filter_edges(depth_canny, depth_gradient)

        return depth_colored, depth_normalized, depth_high_laplacian, depth_canny, edges
    

    def add_stats_on_image(self, camera_stats, imu_stats, latency_ms, image_to_display):
        fps_text = f"FPS: {camera_stats['fps']:.1f} | IMU: {imu_stats['hz']:.1f}Hz | Latency: {latency_ms:.1f}ms"
        dropped_text = f"Dropped: Cam={camera_stats['dropped_frames']} IMU={imu_stats['dropped_packets']}"
        cv2.putText(image_to_display, dropped_text, (10, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        cv2.putText(image_to_display, fps_text, (10, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        return image_to_display