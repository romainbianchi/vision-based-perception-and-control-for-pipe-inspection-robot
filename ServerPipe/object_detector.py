import cv2
import numpy as np
from collections import deque

class ObjectDetector:

    def __init__(self):
        pass

    def detect_obstacle(self, image, edges):
         
        # Coordinates of the pixels with high score
        y_coords, x_coords = np.where(edges == 255)
        
        # No object detected
        if len(y_coords) == 0 or len(x_coords) == 0:
            return False, image, None, None
        
        # Upper left and lower right pixel of the detected high score
        x_low, y_low = max(x_coords), max(y_coords)
        x_high, y_high = min(x_coords), min(y_coords)

        # Bounding Box coordinates
        bounding_box_coords = ((x_high, y_high), (x_low, y_low))
        center_x = x_high + (x_low - x_high) // 2
        center_y = y_high + (y_low - y_high) // 2
        center = (center_y, center_x)

        # Draw green rectangle
        image_with_rectangle = image.copy()
        if image.shape == 2:
            image_with_rectangle = cv2.cvtColor(image_with_rectangle, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(image_with_rectangle, (x_high, y_high), (x_low, y_low), (0, 255, 0), 2)

        return True, image_with_rectangle, bounding_box_coords, center


class JunctionDetector:

    def __init__(self, cam_matrix_path, window_size):
        self.K = self.load_camera_matrix(cam_matrix_path)
        self.window_size = window_size
        initial_window_element = {'detected': 0, 'center': (0, 0), 'radius': 0}
        self.detection_window = deque([initial_window_element.copy() for _ in range(self.window_size)], maxlen=self.window_size)
        self.detection_state = 0 # Flag raised when in junction detected state
        self.last_junction_id = 0 # Identifier of the last junction detected

    def load_camera_matrix(self, cam_matrix_path):
        with open(cam_matrix_path, 'r') as f:
            lines = f.readlines()
        
        P = np.zeros((3, 4))
        for i in range(3):
            P[i] = list(map(float, lines[i].strip().split()))

        K = P[:, :3]
        
        return K

    def detect_circles(self, image):
        gimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        gimage = cv2.GaussianBlur(gimage, (5, 5), sigmaX=2)
        circles = cv2.HoughCircles(gimage,cv2.HOUGH_GRADIENT_ALT,1.5,50,param1=150,param2=0.95,minRadius=5,maxRadius=0)

        if circles is None:
            self.detection_window.append({'detected': 0, 'center': (0,0), 'radius': 0}) # Update junction detection window, no junction detected
            return image, None
        
        image_with_circles = image.copy()
        circles_out = []

        circles = np.uint16(np.around(circles))
        largest_circle_idx = np.argmax(circles[0, :, 2])
        for i, (x, y, r) in enumerate(circles[0, :]):
            if i == largest_circle_idx:
                color = (0, 200, 0)
                draw_center = True
            else:
                color = (0, 40, 0)
                draw_center = False
            circles_out.append({"center": (int(x), int(y)), "radius": int(r)})
            # draw the outer circle
            cv2.circle(image_with_circles,(x,y),r,color,2)
            # draw the center of the circle
            if draw_center:
                cv2.circle(image_with_circles,(x,y),2,(0,0,255),3)

        center_x = circles[0, largest_circle_idx, 0]
        center_y = circles[0, largest_circle_idx, 1]
        radius = circles[0, largest_circle_idx, 2]
        self.detection_window.append({'detected': 1, 'center': (center_x,center_y), 'radius': radius}) # Update junction detection window, junction detected
        
        return image_with_circles, circles_out
    
    def compute_dist_to_junc(self, junction_rad, circle):
        
        rad_px = circle['radius']
        fx = self.K[0,0]
        fy = self.K[1,1]
        f = (fx + fy)/2

        distance = f * junction_rad / rad_px

        return distance
    
    def windowed_junction_detection(self, nb_detect_thresh=8, radius_var_threshold=50):
        junction_detected = [item['detected'] for item in self.detection_window]
        junction_centers = [item['center'] for item in self.detection_window]
        radius = [item['radius'] for item in self.detection_window if item['detected']]

        nb_detection = np.sum(junction_detected)

        if nb_detection == 0:
            return 
        
        radius_var = np.var(radius)

        if self.detection_state == 0:
            if nb_detection > nb_detect_thresh and radius_var < radius_var_threshold:
                self.detection_state = 1 # Junction detected
                self.last_junction_id += 1

        elif self.detection_state == 1:
            if nb_detection < self.window_size - nb_detect_thresh or radius_var > radius_var_threshold:
                self.detection_state = 0 # No junction detected

        print(f'junction number: {self.last_junction_id}')
        print(f'detection state {self.detection_state}')

    def reset_detection(self):
        self.detection_window.clear()
        initial_window_element = {'detected': 0, 'center': (0, 0), 'radius': 0}
        self.detection_window = deque([initial_window_element.copy() for _ in range(self.window_size)], maxlen=self.window_size)
        self.detection_state = 0
        self.last_junction_id = 0
