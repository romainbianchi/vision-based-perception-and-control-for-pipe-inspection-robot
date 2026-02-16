import socket
import threading
import time
from enum import Enum
from collections import deque
import numpy as np

class RobotCommand(Enum):
    STOP = "STOP"
    FORWARD = "UP"
    BACKWARD = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"

class RobotController():

    def __init__(self, udp_ip="192.168.4.1", udp_port=4210, send_interval=0.05):
        self.udp_ip = udp_ip
        self.udp_port = udp_port
        self.send_interval = send_interval

        # Create UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.running = False
        self.control_enabled = False
        self.current_command = RobotCommand.STOP

        # Thread
        self.control_thread = None
        self.command_lock = threading.Lock()

        # Next direction prediction
        self.next_direction = None
        self.direction_window_size = 5
        self.direction_window = deque(maxlen=self.direction_window_size)

        # Flag when robot is in a turn
        self.in_turn = False
        self.turn_start = False
        self.turn_end = False

    def start_communcation(self):
        """Start communcitation"""
        if self.running:
            return
        
        self.running = True
        self.control_thread = threading.Thread(target=self.communication_loop)
        self.control_thread.daemon = True
        self.control_thread.start()

    def stop_communication(self):
        """Stop communication"""
        if not self.running:
            return
        
        self.running = False
        self.send_command(RobotCommand.STOP)
        if self.control_thread:
            self.control_thread.join(timeout=1.0)

    def start(self):
        """Start robot control"""
        self.control_enabled = True
        self.send_command(RobotCommand.FORWARD)

    def stop(self):
        """Stop robot control"""
        self.control_enabled = False
        self.send_command(RobotCommand.STOP)

    def send_command(self, command):
        """Send a command to the robot"""
        try:
            if isinstance(command, RobotCommand):
                command_str = command.value
            else:
                command_str = str(command)

            self.sock.sendto(command_str.encode(), (self.udp_ip, self.udp_port))

            with self.command_lock:
                if isinstance(command, RobotCommand):
                    self.current_command = command

        except Exception as e:
            print(f"Error sending command {command}: {e}")

    def communication_loop(self):
        """Main communication loop"""
        while self.running:

            if not self.control_enabled:
                time.sleep(self.send_interval)
                continue

            with self.command_lock:
                command = self.current_command
            
            if command:
                self.send_command(command)

            time.sleep(self.send_interval)

    def update_direction(self, direction, confidence):
        """Update next direction predicted for the controller"""
        self.direction_window.append((direction, confidence))

        if len(self.direction_window) < self.direction_window_size:
            self.next_direction = None
            return

        # Check over a window of frames for direction validation
        first_class, _ = self.direction_window[0]
        if all(cls == first_class and conf > 90 for cls, conf in self.direction_window) and not self.in_turn:
            self.next_direction = first_class


    def update_command(self, tof_value, tof_low_threshold, tof_high_threshold):
        """Update predicted direction and adapt robot command"""
        # detect end of turn
        if self.in_turn and tof_value > tof_high_threshold:
                print('END OF TURN')
                self.turn_end = True
                self.in_turn = False
                return

        # Adapt low threshold to the type of turn
        if self.next_direction in ['left45', 'right45']:
            tof_low_threshold *= 1.5

        # Start of the turn
        if self.next_direction in ['left45', 'left90', 'right45', 'right90'] and tof_value <= tof_low_threshold:
            print('START TURN')
            self.in_turn = True
            self.turn_start = True

        # Update command
        if self.in_turn:
            if self.next_direction in ['left45', 'left90']:
                self.current_command = RobotCommand.LEFT
            elif self.next_direction in ['right45', 'right90']:
                self.current_command = RobotCommand.RIGHT
        else:
            self.current_command = RobotCommand.FORWARD
    
    def turn_started(self):
        """Check if a turn has started"""
        return self.turn_start
    
    def turn_ended(self):
        """Check if a turn has ended"""
        return self.turn_end
    
    def set_turn_started(self, value):
        """Set the turn started flag"""
        self.turn_start = value

    def set_turn_ended(self, value):
        """Set the turn ended flag"""
        self.turn_end = value
    
