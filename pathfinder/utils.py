from datetime import date
import time

planets = ['mercury', 'venus', 'earth', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune']

# Convert a Unix timestamp to Modified Julian date, epoch 2000 format.
# def unix_to_mjd2000(unix: float):
#     return unix * pk.SEC2DAY - 10957.0
# 
# # Convert an MJD2000 date to a Unix timestamp
# def mjd2000_to_unix(mjd2000: float):
#     return mjd2000 * pk.DAY2SEC + 10957.0
# 
# # Gets the current time as a pk.epoch object
# def current_epoch() -> pk.epoch:
#     return pk.epoch(unix_to_mjd2000(time.time()))
# 
# def epoch_to_isoformat(epoch: pk.epoch) -> str:
#     return date.fromtimestamp(mjd2000_to_unix(epoch.mjd2000)).isoformat()
