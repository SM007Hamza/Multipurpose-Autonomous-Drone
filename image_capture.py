from djitellopy import tello
import cv2

me = tello.Tello()
me.connect()
print(me.get_battery())

me.streamon()

while True:
    img = me.get_frame_read().frame  #to get frame from the drone
    #img = cv2.imread('cat_pic.jpeg', cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (360, 240))
    cv2.imshow("Image", img)
    cv2.waitKey(1)


