from djitellopy import tello, Tello
from time import sleep

me = tello.Tello()
me.connect()
# function to get battery
print(me.get_battery())


# function to get speed
def get_speed(self):
    return self.send_read_command('speed?')


# function to get temperature
def get_temperature(self):
    return self.send_read_command('temperature?')


# function to get altitude
def get_attitude(self):
    return self.send_read_command('attitude?')


# function to get total flight time
def get_flight_time(self):
    return self.send_read_command('time?')


# command for takeoff
me.takeoff()
me.send_rc_control(0, 10, 0, 0)
sleep(2)
me.send_rc_control(0, 0, 0, 30)
sleep(2)
me.send_rc_control(0, 0, 0, 0)
me.land()
