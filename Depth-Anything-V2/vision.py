import socket
import cv2
import numpy as np

FRAME_WIDTH, FRAME_HEIGHT = 320, 240
BYTES_PER_PIXEL = 1
FRAME_SIZE = FRAME_WIDTH * FRAME_HEIGHT * BYTES_PER_PIXEL

def read_frame(client_socket):
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


def read_frame_chunks(client_socket):
    # Read 4-byte header (frame size)
    header = client_socket.recv(4)
    if len(header) != 4:
        return None
    
    frame_size = int.from_bytes(header, byteorder='little')
    if frame_size > (FRAME_HEIGHT*FRAME_WIDTH*4):  # Sanity check
        print(f"Invalid frame size: {frame_size}")
        return None
    
    # Receive frame in chunks
    frame_data = bytearray()
    while len(frame_data) < frame_size:
        try:
            chunk = client_socket.recv(min(1024, frame_size - len(frame_data)))
            if not chunk:
                return None
            frame_data.extend(chunk)
        except socket.timeout:
            print("Timeout waiting for frame data")
            return None
        except ConnectionResetError:
            print("Client disconnected")
            return None
    
    return bytes(frame_data)

def read_ToF_value(client_socket):
    msg = client_socket.recv(2)
    distance = int.from_bytes(msg, byteorder='little')
    return distance


def process_frame(frame_data):
    """Handles both grayscale (1Bpp) and RGB565 (2Bpp) frames"""
    total_bytes = len(frame_data)
    expected_grayscale = FRAME_WIDTH * FRAME_HEIGHT * 1
    expected_rgb565 = FRAME_WIDTH * FRAME_HEIGHT * 2

    if total_bytes == expected_grayscale:
        # Process as grayscale
        image = np.frombuffer(frame_data, dtype=np.uint8)  # uint8 for 1-byte grayscale
        image = np.reshape(image, (FRAME_HEIGHT, FRAME_WIDTH))
        return image # Convert to 3-channel for display
        
    elif total_bytes == expected_rgb565:
        # Process as RGB565
        frame_array = np.frombuffer(frame_data, dtype='>u2')  # uint16 for 2-byte RGB565
        frame_array = np.reshape(frame_array, (FRAME_HEIGHT, FRAME_WIDTH))
        r = ((frame_array & 0xF800) >> 11) * 255 // 31
        g = ((frame_array & 0x07E0) >> 5) * 255 // 63
        b = ((frame_array & 0x001F) >> 0) * 255 // 31
        image = np.stack([r, g, b], axis=-1).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image
        
    else:
        print(f"Unexpected frame size: {total_bytes} bytes (Expected: {expected_grayscale} or {expected_rgb565})")
        return None


def flip_image(image):
    image = np.flipud(image)
    image = np.fliplr(image)
    return image


def get_image(client_socket):

    frame = read_frame(client_socket)

    if frame is None:
        return None
    
    image = process_frame(frame)
    
    return image


def add_fps_text(image, fps):
    # Convert to 3-channel if needed
    if len(image.shape) == 2:
        display_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        display_image = image.copy()
    
    # Add FPS text
    fps_text = f"FPS: {fps:.2f}"
    display_image = cv2.putText(display_image, fps_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                (0, 255, 0), 2)
    
    # Put the display image in the queue
    return display_image
