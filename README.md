# Multipurpose Autonomous Drone

This project implements functionalities that are required in an Autonomous drone. We use a Quadcopter drone from DJI Tello. DJI Tello has Open Source SDK which we use in this project to implement various features. Some of the featues that we implement are as follows:

- object deteection
- object avoidance
- path following
- face tracking
- fail-safe landing
- telemetry emission
- remote command and code deployment

## Requirements

- DJI Tello Drone
- Computer with WLAN (WiFi 802.11ac) support

## Setup instructions

To run this project, a DJI Tello drone is required. We also need to setup the server, which the drone communicates with and depends on for various compute extensive tasks such as object detection, face tracking, among others. Features like fail-safe landing and object avoidance run in the drone itself, so that in-case of connectivity issues, drone can land safely.

### Drone Setup

Make sure drone's firmware is updated to the latest version. Also, make sure drone is charged above 80% and connected to power before starting firmware update. Once firmware has been updated, switch on the drone and then configure the server.

### Server Setup

Create a virtual environment and install requirements.txt there. Then initialize the virtual environment.
Next turn on wifi and search for drone's network-id: **DJI Tello**
Drone talks to the server over encrypted UDP channel and connecting laptop/computer to drone's WiFi enables this transmission.

---

Following are the modules and their usage:

_basic_drone_movement.py_: It sets the drone in flight based on the passed coordinates and continuosly sends telemetry data to the server.

_object_detection.py_: It employs computer vision to detect objects on the streamed frames from drone's camera.

_face_tracking.py_: It is used for tracking subject's face. The model is trained with human faces but can be easily changed to also track wild animals. When drone enters into this mode, it keeps hovering near the subject with a safe distance and moves as the subject moves.

_mapping.py_: This module is used to map and trace the movement of drone while it is in flight. This is not only used for telemetry purposes but can also be used to future training.

_keyboard_control_surveillance.py_: This module is used to record the video captured by the drone. The video is streamed to the server where the video is saved. Drone donesn't have enough persistent storage to save streamed video which is 720p at 30 frames per second.

_collisionAvoidance_: This module is more complex than other modules as it requires more extensive traning with our own data and based on the simulated environment. Once the code is deployed with the trained model, it can avoid objects. This feature is also used for fail-safe landing.

_safe_landing.py_: This module is used to make sure drone lands safely in-case the battery is below a certain threshold or it detects objects which it cannot avoid.

_line_follower.py_: When the drone enter this mode, it uses contour and pattern detection to follow a line as detected by it's camera.
