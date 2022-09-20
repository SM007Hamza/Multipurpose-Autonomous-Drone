from dronekit import connect

# Connecting the Vehicle
vehicle = connect('udpin:127.0.0.1:14551', baud=115200)

# Printing Vehicle's Latitude
print("Vehicle's Latitude              =  ", vehicle.location.global_relative_frame.lat)

# Printing Vehicle's Longitude
print("Vehicle's Longitude             =  ", vehicle.location.global_relative_frame.lon)