from tensorflow.keras.models import Model
from tensorflow.keras.layers import LeakyReLU
import tensorflow.keras.backend as K
from io_project.read_utils import generateXandY
from constants.AI_params import *
from config.MainConfig import get_prediction_params
from models.modelSelector import select_2d_model
from models.model_viz import print_layer_names, plot_intermediate_2dcnn_feature_map, plot_cnn_filters_by_layer
from constants.AI_params import TrainingParams
from constants_proj.AI_proj_params import PredictionParams, ProjTrainingParams, NetworkTypes
import matplotlib.image as mpltimage
from inout.io_netcdf import read_netcdf
from models_proj.models import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from os.path import join

if __name__ == '__main__':
    config = get_prediction_params()
    input_folder = config[PredictionParams.input_folder]
    rows = config[ProjTrainingParams.rows]
    cols = config[ProjTrainingParams.cols]
    field_names = config[ProjTrainingParams.fields_names]
    obs_field_names = config[ProjTrainingParams.fields_names_obs]
    output_fields = config[ProjTrainingParams.output_fields]
    run_name = config[TrainingParams.config_name]
    output_folder = join(config[PredictionParams.output_imgs_folder], 'MODEL_VISUALIZATION', run_name)
    norm_type = config[ProjTrainingParams.norm_type]

    model_weights_file = config[PredictionParams.model_weights_file]

    net_type = config[ProjTrainingParams.network_type]
    if net_type == NetworkTypes.UNET or net_type == NetworkTypes.UNET_MultiStream:
        model = select_2d_model(config, last_activation=None)
    if net_type == NetworkTypes.SimpleCNN_2:
        model = simpleCNN(config, nn_type="2d", hid_lay=2, out_lay=2)
    if net_type == NetworkTypes.SimpleCNN_4:
        model = simpleCNN(config, nn_type="2d", hid_lay=4, out_lay=2)
    if net_type == NetworkTypes.SimpleCNN_8:
        model = simpleCNN(config, nn_type="2d", hid_lay=8, out_lay=2)

    print('Reading weights ....')
    model.load_weights(model_weights_file)

    # # Gets all the config
    model_config = model.get_config()

    # # All Number of parameters
    print(F' Number of parameters: {model.count_params()}')
    # Number of parameters by layer
    print(F' Number of parameters first CNN: {model.layers[1].count_params()}')

    # Example of plotting the filters of a single layer
    print("Printing layer names:")
    print_layer_names(model)
    # plot_cnn_filters_by_layer(model.layers[1], 'First set of filters')  # The harcoded 1 should change by project

    # *********** Read files to predict***********
    # # ========= Here you need to build your test input different in each project ====
    all_files = os.listdir(input_folder)
    all_files.sort()
    model_files = np.array([x for x in all_files if x.startswith('model')])
    increment_files = np.array([x for x in all_files if x.startswith('increment')])
    obs_files = np.array([x for x in all_files if x.startswith('obs')])

    # ========= Here you need to build your test input different in each project ====
    all_files = os.listdir(input_folder)
    all_files.sort()
    model_files = np.array([x for x in all_files if x.startswith('model')])

    for id_file, c_file in enumerate(model_files):
        # Find current and next date
        year = int(c_file.split('_')[1])
        day_of_year = int(c_file.split('_')[2].split('.')[0])

        model_file = join(input_folder, F'model_{year}_{day_of_year:03d}.nc')
        inc_file = join(input_folder,F'increment_{year}_{day_of_year:03d}.nc')
        obs_file = join(input_folder,F'obs_{year}_{day_of_year:03d}.nc')

        # *********************** Reading files **************************
        z_layers = [0]
        input_fields_model = read_netcdf(model_file, field_names, z_layers)
        input_fields_obs = read_netcdf(obs_file, obs_field_names, z_layers)
        output_field_increment = read_netcdf(inc_file, output_fields, z_layers)

        # ******************* Normalizing and Cropping Data *******************
        for start_row in np.arange(0, 891-rows, rows):
            for start_col in np.arange(0, 1401-cols, cols):
                try:
                    input_data, y_data = generateXandY(input_fields_model, input_fields_obs, output_field_increment, field_names, obs_field_names, output_fields,
                                                       start_row, start_col, rows, cols, norm_type=norm_type)
                except Exception as e:
                    print(F"Failed for {c_file} row:{start_row} col:{start_col}")
                    continue

                X_nan = np.expand_dims(input_data, axis=0)
                Y_nan = np.expand_dims(y_data, axis=0)

                # ******************* Replacing nan values *********
                # We set a value of 0.5 on the land. Trying a new loss function that do not takes into account land
                X = np.nan_to_num(X_nan, nan=0)
                Y = np.nan_to_num(Y_nan, nan=-0.5)

                output_nn = model.predict(X, verbose=1)
                output_nn[np.isnan(Y_nan)] = np.nan
                # =========== Output from the last layer (should be the same as output_NN
                print("Evaluating all intermediate layers")
                inp = model.input # input placeholder
                outputs = [layer.output for layer in model.layers[1:] if layer.name.find("conv") != -1]  # Displaying only conv layers
                # All evaluation functions (used to call the model up to each layer)
                functors = [K.function([inp], [out]) for out in outputs]
                # Outputs for every intermediate layer
                layer_outs = [func([X]) for func in functors]

                for layer_to_plot in range(0, len(outputs)-1):
                    title = F'Layer {layer_to_plot}_{outputs[layer_to_plot].name}. {c_file}_{start_row:03d}_{start_col:03d}'
                    file_name = F'{c_file}_{start_row:03d}_{start_col:03d}_lay_{layer_to_plot}'
                    plot_intermediate_2dcnn_feature_map(layer_outs[layer_to_plot][0],
                                                        input_data=X_nan,
                                                        desired_output_data=Y_nan,
                                                        nn_output_data=output_nn,
                                                        input_fields=input_fields_model+input_fields_obs,
                                                        title=title,
                                                        output_folder=output_folder,
                                                        file_name=file_name, disp_images=False)
