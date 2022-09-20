import cv2
from djitellopy import tello
import cvzone

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 400)

while True:
    success, img = cap.read()
    cv2.imshow("Image", img)
    cv2.waitKey(1)
"""
This project is an Autonomous drone which has features like object deteection , object avoidance , path following , face tracking and fail safe
(landing)

Install requirements.txt

This project requires DJI Tello drone

run basic_drone_movement
It will fly by its coordinates and give telemetry_data to thee mother
(station)

run object_detection.py to track the objects through the drone

run face_tracking.py to make the drone autonomously fly detecting the face or person or animal

run mapping.py to map the drone directions while flying

run keyboard_control_surveillance to record the video by manually controlling through the
(keyboard)
To run collisionAvoidance follow the following steps

Run PyDNet on Tello feed

To run pydnet, just launch

python3 tello_pydnet_interface.py --checkpoint_dir /checkpoint/IROS18/pydnet --resolution [1,2,3]
Please note that the velocity commands have been commented out. You could either uncomment them or create your own navigation algorithm.

Train PyDNet from Scratch
monodepth (https://github.com/mrharicot/monodepth) framework by Clément Godard
After you have cloned the monodepth repository, add to it the scripts contained in training_code folder from this repository (you have to replace the original monodepth_model.py script). Then you can train pydnet inside monodepth framework.

Evaluate PyDNet on Eigen split
To get results on the Eigen split, just run
python3 experiments.py --datapath PATH_TO_KITTI --filenames PATH_TO_FILELIST --checkpoint_dir /checkpoint/IROS18/pydnet --resolution [1,2,3]
run safe_landing.py to make the drone land safe without crashing.

"""