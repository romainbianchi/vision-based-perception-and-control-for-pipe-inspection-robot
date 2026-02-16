import cv2
import numpy as np
import open3d as o3d


def scale_depth_map(depth, tof_value, obj_center):
    height, width = depth.shape
    img_center = (height // 2, width // 2)

    if obj_center is not None:
        center = obj_center
    else:
        center = img_center

    try:
        center_pixel = depth[center]
    except:
        print('object center out of bounds, impossible to scale depth map')
        return None

    if center_pixel == 0:
        print("Center depth value is zero. Cannot scale")
        return depth.copy()
    
    scale = tof_value / center_pixel

    scaled_depth = depth * scale

    return scaled_depth


def normalize_and_colormap(image):
    image_normalized_0_255 = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    image_normalized = cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX)
    image_uint8 = image_normalized_0_255.astype(np.uint8)
    image_colored = cv2.applyColorMap(image_uint8, cv2.COLORMAP_JET)

    return image_normalized, image_colored


def get_depth_gradient(depth):
    # Compute the gradient of the depth map
    gradient_x = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=5)
    gradient_y = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=5)

    # Compute the magnitude of the gradient
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    return gradient_magnitude


def remove_far_elements(image, depth, std_fact = 0.2):
    # Remove elements that are far from the camera
    mean = np.mean(depth)
    std = np.std(depth)
    foreground_mask = depth > (mean + std_fact * std)
    # Apply mask to the binary depth gradient image
    image_front = np.zeros_like(image)
    image_front[foreground_mask] = image[foreground_mask]

    return image_front


def large_gradient(gradient, threshold, ratio):
    # Compute thresholds
    fixed_threshold = threshold
    dynamic_threshold = ratio * np.max(gradient)
    
    # Create condition mask: only keep values satisfying both conditions
    condition = np.logical_and(gradient > dynamic_threshold, gradient > fixed_threshold)
    
    # Create output array: 255 where condition is True, else 0
    output = np.zeros_like(gradient, dtype=np.uint8)
    output[condition] = 255
    
    return output


def dist_grad_score(depth, gradient, alpha = 0.5, beta = 0.5):
    max_depth = np.max(depth)
    max_grad = np.max(gradient)

    score = (alpha / max_depth) * depth + (beta / max_grad) * gradient

    max_score = np.max(score)
    min_score = np.min(score)

    return score


def large_score(score, threshold = 0.5):
    
    # Create condition mask: only keep values satisfying both conditions
    condition = score > threshold
    
    # Create output array: 255 where condition is True, else 0
    output = np.zeros_like(score, dtype=np.uint8)
    output[condition] = 255
    
    return output


def detect_obstacle(image, depth, depth_gradient_front, tol=0.2):
    # Coordinates of the lowest pixel with 255 value
    y_coords, x_coords = np.where(depth_gradient_front == 255)

    # No object detected
    if len(y_coords) == 0 or len(x_coords) == 0:
        print("No object detected")
        return False, None, None, None

    x_low, y_low = max(x_coords), max(y_coords)
    x_high, y_high = min(x_coords), min(y_coords)

    window_half_size = 7
    x_start = max(0, x_low - window_half_size)
    x_end = min(depth.shape[1], x_low + window_half_size + 1)
    y_start = max(0, y_low - window_half_size)
    y_end = min(depth.shape[0], y_low + window_half_size + 1)

    window = depth[y_start:y_end, x_start:x_end]
    max_depth_value_window = np.round(np.max(window), 0)
    min_depth_value_window = np.min(window)
    rel_y, rel_x = np.unravel_index(np.argmin(window), window.shape)
    
    # Convert to image coordinates
    min_x = x_start + rel_x
    min_y = y_start + rel_y

    # Vectorized search for the first pixel in depth column min_x that matches max value
    column = depth[min_y:, min_x]  # Slice the column starting from min_y
    close_match = np.where(np.abs(column - max_depth_value_window) <= tol)[0]  # Find the first match
    
    # If there's no match, np.argmax will return 0, so we check if it's valid
    if close_match.size > 0:
        final_y = min_y + close_match[-1]
    else:
        final_y = depth.shape[0] - 1  # If no max found, use the bottom of the image
    y_low = final_y

    # Draw green rectangle
    image_with_rectangle = image.copy()
    image_with_rectangle = cv2.cvtColor(image_with_rectangle, cv2.COLOR_GRAY2BGR)
    cv2.circle(image_with_rectangle, (min_x, min_y), 2, (0, 0, 255))
    cv2.rectangle(image_with_rectangle, (x_high, y_high), (x_low, y_low), (0, 255, 0), 2)

    # Bounding Box coordinates
    coords = ((x_high, y_high), (x_low, y_low))
    center_x = x_high + (x_low - x_high) // 2
    center_y = y_high + (y_low - y_high) // 2
    center = (center_y, center_x)

    # Draw center
    cv2.circle(image_with_rectangle, center[::-1], 3, (0, 255, 0))

    return True, image_with_rectangle, coords, center


def estimate_obj_height(obstacle_bounding_box, distance, K, cam_height):

    # Image plane coordinates
    y_top_pix = obstacle_bounding_box[0][1]
    y_bottom_pix = obstacle_bounding_box[1][1]

    # Camera intrinsic parameters
    f_y = K[1,1]
    c_y = K[1,2]

    # Convert to world coordinates
    y_top = cam_height + distance * ((y_top_pix - c_y) / f_y)
    y_bottom = cam_height + distance * ((y_bottom_pix - c_y) / f_y)
    
    height = np.abs(y_top - y_bottom)

    return height