# Vision-Based Perception and Control for a Pipe Inspection Robot

This repository contains the code developed for a vision-based perception and control system for a pipe inspection robot.

The project demonstrates how a robot equipped with a **monocular camera, and time-of-flight sensor** can navigate inside pipe networks, estimate its position, and detect obstructions using computer vision.

<br>
<br>

<p align="center">
  <img src="images/Robot.png" alt="Robot" height="300"/>
  <img src="images/Detection.png" alt="Detection" height="300"/>
</p>

<p align="center">
  <img src="images/Classification.png" alt="Classification" width="760"/>
</p>

## System Architecture

The system consists of:

- **Nicla Vision** mounted on the robot (camera, ToF sensor), which captures images and distance measurements.

- A **Python server running on a laptop** that receives the sensor data and runs the main perception and navigation algorithms.

- An **ESP32 motor controller** that receives motion commands from the laptop and controls the robot motors.

<br>

<p align="center">
  <img src="images/SoftwareArchitecture.png" alt="SoftwareArchitecture" width="760"/>
</p>

## Main Features

### Pipe Turn Detection and Navigation

A deep learning classifier predicts the direction of upcoming pipe turns (left/right, 45° or 90°).  
Using this prediction together with distance measurements, the robot performs differential control to successfully navigate tight turns.

### Junction Detection and Mapping

Pipe junctions are detected using computer vision (Hough circle detection).  
These detections allow the system to:

- Estimate the distance to junctions
- Update the robot position
- Incrementally build a topological map of the pipe network

### Obstruction Detection

A monocular depth estimation model (Depth-Anything-V2: https://github.com/DepthAnything/Depth-Anything-V2) is used to detect objects blocking the pipe.  
Depth gradients are used to identify object contours and generate bounding boxes around potential obstructions.

## Project Context

This project was developed as part of a **Master's thesis in Robotics at EPFL** within the **Computational Robot Design & Fabrication Lab**.
