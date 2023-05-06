
RAW_DATA_FORMAT = {
    "TIMESTAMP": 0,
    "TRACK_ID": 1,
    "OBJECT_TYPE": 2,
    "X": 3,
    "Y": 4,
    "CITY_NAME": 5,
}

LANE_WIDTH = {'MIA': 3.84, 'PIT': 3.97}
VELOCITY_THRESHOLD = 1.0
# Number of timesteps the track should exist to be considered in social context
EXIST_THRESHOLD = 50
# index of the sorted velocity to look at, to call it as stationary
STATIONARY_THRESHOLD = 13
color_dict = {"AGENT": "#FF0000", "OTHERS": "#00FFFF", "AV": "#00FF00"}
LANE_RADIUS = 30
OBJ_RADIUS = 30
DATA_DIR = "./data"
OBS_LEN = 20
INTERMEDIATE_DATA_DIR = './interm_data'

