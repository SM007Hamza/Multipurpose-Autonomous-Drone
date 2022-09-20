from djitellopy import tello


me = tello.Tello()
me.connect()

print(me.get_battery())


def get_speed(self):
    return self.send_read_command('speed?')


def get_temperature(self):
    return self.send_read_command('temperature?')


def get_attitude(self):
    return self.send_read_command('attitude?')


def get_flight_time(self):
    return self.send_read_command('time?')
