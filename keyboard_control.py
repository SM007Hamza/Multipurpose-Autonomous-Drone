from djitellopy import tello
import key_Press_Module as Kp
from time import sleep

Kp.init()
me = tello.Tello()
me.connect()
print(me.get_battery())

# function to Control the drone with the help of the keys.


def getKeyboardInput():
    lr, fb, ud, yv = 0, 0, 0, 0
    speed = 50
    if Kp.getKey("LEFT"):
        lr = -speed
    elif Kp.getKey("RIGHT"):
        lr = speed

    if Kp.getKey("UP"):
        fb = speed
    elif Kp.getKey("DOWN"):
        fb = -speed

    if Kp.getKey("w"):
        ud = speed
    elif Kp.getKey("s"):
        ud = -speed

    if Kp.getKey("a"):
        yv = speed
    elif Kp.getKey("d"):
        yv = -speed

    if Kp.getKey("q"):
        me.land()
    if Kp.getKey("e"):
        me.takeoff()

    return [lr, fb, ud, yv]


while True:
    vals = getKeyboardInput()
    me.send_rc_control(vals[0], vals[1], vals[2], vals[3])
    sleep(0.05)
