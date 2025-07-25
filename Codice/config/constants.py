import os
import platform
def get_home_directory():
    if platform.system() in 'Linux':
        return r'/mnt/c/Users/marco/Desktop/Tesi'
    else:
        return r'C:/Users/marco/Desktop/Tesi'

HOME = get_home_directory()   # <---- CHANGE THIS

DATA_HOME = HOME + r'/data_map03_0911_test'

HOME_SAVING = HOME + r'/Savings'

HOME_DIR = HOME + r'/Codice'

NUMBER_OF_SENSORS = 12

LIDAR_X_DIRECTORY = DATA_HOME + r'/1_original/lidar_output_lidarX/lidar'
LIDAR_X_GRID_DIRECTORY = DATA_HOME + r'/3_lidar_grid_maps/lidarX'

SNAPSHOT_DIRECTORY = DATA_HOME + r'/1_original/positionX'
LIDAR_DIRECTORY = DATA_HOME + r'/1_original'

SNAPSHOT_X_DIRECTORY = DATA_HOME + r'/2_syncronized_snapshot/snapshotX'
SNAPSHOT_X_GRID_DIRECTORY = DATA_HOME + r'/4_snapshot_grid_maps/snapshotX'

SIMULATION_DIR = HOME_DIR + r'/simulation_data'
CONFIG_DIR = HOME_DIR + r'/config'

FFCV_DIR = DATA_HOME + r'/FFCV'

SCALER_DIR = HOME_SAVING + r'/scalers'

MODEL_DIR = HOME_SAVING + r'/models_pred'

# For save_positions
NEW_POSITIONS_OFFSETS = [(70.15, 11.50, -6.0), (100.59, -27.46, -6.0), (96.83, 6.31, -6.0), (65.3, -0.55, -6.0), (73.50, -9.25, -6.0), (85.65, -27.85, -6.0), (29.11, -5.86, -6.0), (5.97, -28.63, -6.0), (-16.59, -1.22, -6.0), (68.46, 128.17, -6.0), (115.36, 129.5, -6.0), (15.34, -125.4, -6.0)]

# For Show_grid_map
MIN_HEIGHT = -60.0

# For Conversion_BB_into_2,5D
REDUCING_RANGE = 0.5 # It can't be set too low because can be generated a bounnfing box with only two points (because we also want the bounging box of cars half out the grid)
INCREASE_GRID_RANGE = 200

# For Conversion_3D_to_2,5D and Conversion_BB_into_2,5D
GRID_RESOLUTION = 0.1
X_RANGE = 400   #Used also in show_grid_map
Y_RANGE = 400   #Used also in show_grid_map
X_MIN = 20      #Used also in show_grid_map
Y_MIN = 20      #Used also in show_grid_map
FLOOR_HEIGHT = -100
RANGE_FOR_ROTATED_VEHICLES = 0.1 # To identify the bounding box of rotated vehicles that are partially outside the grid

# For Show_lidar_data and Show_grid_map
LIDAR_FILE_X = [60, 100, 562, 50, 10, 70, 10, 10, 10, 10, 10, 10]

# For show_grid_and_BB and Conversion_BB_into_2,5D
INCREMENT_BB_PEDESTIAN = 0.12

# For Eliminate_BB_without_points
NUM_MIN_POINTS_VEHICLE = 5
NUM_MIN_POINTS_BICYCLE = 3
NUM_MIN_POINTS_PEDESTRIAN = 1
HEIGHT_OFFSET = 3 # Height from where consider point in the bounding boxes

# For generate_augmentation
AUGMENTATION_FACTOR = 0.3

# For Generate_chuncks
MINIMUM_NUMBER_OF_RILEVATIONS = 1 # Minimum number of past rilevations to consider a bounding box valid

# For Neural_network
SEED = 42
TEST_SIZE = 0.1
NUMBER_OF_CHUNCKS = 15
NUMBER_OF_CHUNCKS_TEST = 2
BATCH_SIZE = 4

NUMBER_RILEVATIONS_INPUT = 1 #Number of rilevations used as input to the model
FUTURE_TARGET_RILEVATION = [0] # How many step in the future to predict respect the last input rilevation (if 1 we predict the rilevation immediatly after the last one)
SELECTED_FUTURE_INDEX = 0 # Index of the future rilevation to use as target (if 0 we use the future rilevation corresponding to the last input rilevation)

# For Diffusion model
MINIMUM_BETHA = 1e-1
MAXIMUM_BETHA = 0.5
RANGE_TIMESTEPS = 1000