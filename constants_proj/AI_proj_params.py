from enum import Enum


class AnalizeDataParams(Enum):
    input_folder_tsis = 1  # Input folder where the DA output is
    input_folder_forecast = 2  # Input folder where the free forecast output is
    input_folder_obs = 3  # Input folder where the observations output is
    output_folder = 4  # Where to output the data
    PIES = 5  # array of options to include PIES or not
    YEARS = 6  # Array with the years to be analyzed
    MONTHS = 7  # Array with the months to be analyzed
    fields_names = 8  # Array with the names of the fields to be analyzed
    fields_names_obs = 9   # Array with the names of the fields in the observation data to be analyzed
    plot_modes_per_field = 10  # How to plot each field (contour or raster)
    layers_to_plot = 11  # Which Z-axis layers to plot

class ParallelParams(Enum):
    NUM_PROC = 1

class ProjTrainingParams(Enum):
    input_folder_tsis = 1  # Input folder where the DA output is
    input_folder_forecast = 2  # Input folder where the free forecast output is
    input_folder_obs = 3  # Input folder where the observations output is
    output_folder = 4  # Where to output the data
    PIES = 5  # array of options to include PIES or not
    YEARS = 6  # Array with the years to be analyzed
    MONTHS = 7  # Array with the months to be analyzed
    fields_names = 8  # Array with the names of the fields to be analyzed
    fields_names_obs = 9   # Array with the names of the fields in the observation data to be analyzed
    output_field_name = 10  # String containing the name of the output field

