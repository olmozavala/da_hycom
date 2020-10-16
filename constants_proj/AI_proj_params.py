from enum import Enum

MAX_MODEL = {'temp': 35, 'srfhgt': 10, 'salin': 40, 'u-vel.': 3, 'v-vel.': 3, 'LAT': 90 , 'LON': 180}
MIN_MODEL = {'temp': 10, 'srfhgt': -10, 'salin': 0, 'u-vel.':-3, 'v-vel.':-3, 'LAT':-90 , 'LON':-180}

MAX_OBS = {'sst': 40, 'ssh': 0.9, 'sss': 40}
MIN_OBS = {'sst': 0,  'ssh':-0.9, 'sss': 0}

MAX_INCREMENT = {'temp': 5, 'srfhgt': 10, 'salin': 1, 'u-vel.': 0.06, 'v-vel.': 0.06, 'LAT': 90, 'LON': 180 }
MIN_INCREMENT = {'temp':-5, 'srfhgt':-10, 'salin':-1, 'u-vel.':-0.06, 'v-vel.':-0.06, 'LAT':-90, 'LON':-180 }


class PreprocParams(Enum):
    input_folder_tsis = 1  # Input folder where the DA output is
    input_folder_hycom = 2  # Input folder where the free forecast output is
    input_folder_obs = 3  # Input folder where the observations output is
    output_folder = 4  # Where to output the data
    imgs_output_folder = 40  # Where to output the imgs
    YEARS = 6  # Array with the years to be analyzed
    MONTHS = 7  # Array with the months to be analyzed
    fields_names = 8  # Array with the names of the fields to be analyzed
    fields_names_obs = 9   # Array with the names of the fields in the observation data to be analyzed
    plot_modes_per_field = 10  # How to plot each field (contour or raster)
    layers_to_plot = 11  # Which Z-axis layers to plot


class ParallelParams(Enum):
    NUM_PROC = 1


class ProjTrainingParams(Enum):
    input_folder_preproc = 20
    output_folder = 4  # Where to output the data
    PIES = 5  # array of options to include PIES or not
    YEARS = 6  # Array with the years to be analyzed
    MONTHS = 7  # Array with the months to be analyzed
    fields_names = 8  # Array with the names of the fields to be analyzed
    fields_names_obs = 9   # Array with the names of the fields in the observation data to be analyzed
    output_fields = 10  # String containing the name of the output field
    prediction_time = 11  # Number of days to make the prediction for
    rows = 12 # The number of rows we will tak from the whole images for training and everything
    cols = 13 # The number of columns we will tak from the whole images for training and everything

class PredictionParams(Enum):
    input_folder = 1  # Where the images are stored
    output_folder = 2  # Where to store the segmented contours
    output_imgs_folder = 3  # Where to store intermediate images
    output_file_name = 4  # Name of the file with the final statistics
    show_imgs = 5  # If we want to display the images while are being generated (for PyCharm)
    model_weights_file = 8  # Which model weights file are we going to use
    # Indicates that we need to resample everything to the original resolution. If that is the case
    metrics = 10
    compute_metrics = 12  # This means we have the GT ctrs
    save_imgs = 16  # Indicates if we want to save images from the segmented contours

