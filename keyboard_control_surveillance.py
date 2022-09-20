from djitellopy import tello
import key_Press_Module as Kp
import time
import cv2


Kp.init()
me = tello.Tello()
me.connect()
print(me.get_battery())
global img

me.streamon()


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
        time.sleep(3)
    if Kp.getKey("e"):
        me.takeoff()

    if Kp.getKey('z'):

        cv2.imwrite(f'/Resources/Images/{time.time()}.jpg', img)

    return [lr, fb, ud, yv]


while True:
    vals = getKeyboardInput()
    me.send_rc_control(vals[0], vals[1], vals[2], vals[3])
    img = me.get_frame_read().frame
    img = cv2.resize(img, (360, 240))
    cv2.imshow("Image", img)
    cv2.waitKey(1)
