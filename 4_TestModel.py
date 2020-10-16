import os
from io_project.read_utils import generateXandY

from numpy.distutils.system_info import flame_info
from tensorflow.keras.utils import plot_model
from inout.io_netcdf import read_netcdf
from os.path import join, exists
import numpy as np
import numpy.ma as ma
import pandas as pd

from img_viz.eoa_viz import EOAImageVisualizer
from config.MainConfig import get_prediction_params
from constants_proj.AI_proj_params import PredictionParams, ProjTrainingParams
from constants.AI_params import TrainingParams
from models.modelSelector import select_2d_model
from models_proj.models import simpleCNN
from img_viz.constants import PlotMode

from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt


def main():
    config = get_prediction_params()
    test_model(config)

def test_model(config):
    input_folder = config[PredictionParams.input_folder]
    output_folder = config[PredictionParams.output_folder]
    output_fields = config[ProjTrainingParams.output_fields]
    model_weights_file = config[PredictionParams.model_weights_file]
    output_imgs_folder = config[PredictionParams.output_imgs_folder]
    field_names = config[ProjTrainingParams.fields_names]
    obs_field_names = config[ProjTrainingParams.fields_names_obs]
    rows = config[ProjTrainingParams.rows]
    cols = config[ProjTrainingParams.cols]
    run_name = config[TrainingParams.config_name]

    output_imgs_folder = join(output_imgs_folder, run_name)

    # *********** Chooses the proper model ***********
    print('Reading model ....')
    # model = select_2d_model(config, last_activation='relu')
    model = simpleCNN(config)
    plot_model(model, to_file=join(output_folder, F'running.png'), show_shapes=True)

    # *********** Reads the weights***********
    print('Reading weights ....')
    model.load_weights(model_weights_file)

    # *********** Read files to predict***********
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
        # for start_row in np.arange(0, 891-rows, rows):
        #     for start_col in np.arange(0, 1401-cols, cols):
        for start_row in np.arange(0, 891-rows, 200):
            for start_col in np.arange(0, 1401-cols, 100):
                try:
                    input_data, y_data = generateXandY(input_fields_model, input_fields_obs, output_field_increment, field_names, obs_field_names, output_fields,
                                                       start_row, start_col, rows, cols)
                except Exception as e:
                    print(F"Failed for {c_file} row:{start_row} col:{start_col}")
                    continue

                X = np.expand_dims(input_data, axis=0)
                Y = np.expand_dims(y_data, axis=0)

                # ******************* Replacing nan values *********
                # We set a value of 0.5 on the land. Trying a new loss function that do not takes into account land
                X = np.nan_to_num(X, nan=0)
                Y = np.nan_to_num(Y, nan=-0.5)

                output_nn_all_norm = model.predict(X, verbose=1)
                land_indexes = Y == -0.5
                ocean_indexes = np.logical_not(land_indexes)
                output_nn_all_norm[land_indexes] = np.nan
                final_output_cnn = np.squeeze(output_nn_all_norm)
                if len(final_output_cnn.shape) == 2: # In this case we only had one output and we need to make it 'array' to plot
                    final_output_cnn = np.expand_dims(final_output_cnn, axis=2)

                mse_cnn = np.sqrt(mean_squared_error(output_nn_all_norm[ocean_indexes], Y[ocean_indexes]))
                #
                # errors_df.iloc[id_file] = [mse_pers, mse_cnn, mse_cnn_pers]
                # # ================== Making plots ==========
                # titles = [F"$TSIS^{{t}}$",
                #           F"$TSIS^{{t+{day_to_predict}}}$",
                #           F"$CNN^{{t+{day_to_predict}}}$"]
                #
                # fields = [persistence_preproc[0, :, :, 0],
                #           Y[0, :, :, 0],
                #           output_nn_all_norm[0, :, :, 0]]
                #

                b_size = 6 # Boundary size we don't want to take into account

                # viz_obj = EOAImageVisualizer(output_folder=output_imgs_folder, disp_images=False, mincbar=mincbar, maxcbar=maxcbar)
                viz_obj = EOAImageVisualizer(output_folder=output_imgs_folder, disp_images=False)
                # ================== This displays all inputs and the cnn output ================
                viz_obj.plot_2d_data_np_raw(np.concatenate((input_data.swapaxes(0,2), y_data.swapaxes(0,2))),
                # viz_obj.plot_2d_data_np_raw(np.concatenate((X[0,:,:,:].swapaxes(0,2), Y[0,:,:,:].swapaxes(0,2))),
                                            var_names=[F"in_model_{x}" for x in field_names] +
                                                      [F"in_obs_{x}" for x in obs_field_names] +
                                                      [F"out_inc_{x}" for x in output_fields],
                                            file_name=F"Input_and_CNN_{c_file}_{start_row:03d}_{start_col:03d}",
                                            rot_90=True,
                                            title=F"Input data: {field_names} and {obs_field_names}, output {output_fields}")

                # ================== Displays desired output ================
                viz_obj.output_folder = join(output_imgs_folder,'expected')
                viz_obj.plot_2d_data_np_raw(y_data.swapaxes(0,2)[:, b_size:-b_size, b_size:-b_size],
                                            var_names= [F"out_inc_{x}" for x in output_fields],
                                            file_name=F"Expected_output_{c_file}_{start_row:03d}_{start_col:03d}",
                                            rot_90=True,
                                            title=F"Expected {output_fields}")

                # ================== Displays only CNN output ================
                viz_obj.output_folder = join(output_imgs_folder,'cnn')
                viz_obj.plot_2d_data_np_raw(final_output_cnn.swapaxes(0,2)[:, b_size:-b_size, b_size:-b_size],
                                            var_names= [F"out_inc_{x}" for x in output_fields],
                                            file_name=F"Only_CNN_{c_file}_{start_row:03d}_{start_col:03d}",
                                            rot_90=True,
                                            title=F"Output {output_fields}")

                # ================== Displays CNN error ================
                viz_obj.output_folder = join(output_imgs_folder,'error')
                viz_obj.plot_2d_data_np_raw((y_data - final_output_cnn).swapaxes(0,2)[:, b_size:-b_size, b_size:-b_size],
                                            var_names= [F"out_inc_{x}" for x in output_fields],
                                            file_name=F"ERROR_CNN_{c_file}_{start_row:03d}_{start_col:03d}",
                                            rot_90=True,
                                            title=F"RMSE {output_fields}  {mse_cnn}")

                # titles = [F"PERSISTENCE \n $TSIS^{{t+{day_to_predict}}} - TSIS^{{t}} $ MSE ~{mse_pers:0.4f}",
                #           F"CNN \n $TSIS^{{t+{day_to_predict}}} - CNN^{{t+{day_to_predict}}}$ MSE ~{mse_cnn:0.4f}",
                #           F"$TSIS^{{t}} - CNN^{{t+{day_to_predict}}}$ MSE ~{mse_cnn_pers:0.4f}"]
                #
                # fields = [Y[0, :, :, 0] - persistence_preproc[0, :, :, 0],
                #           Y[0, :, :, 0] - output_nn_all_norm[0, :, :, 0],
                #           persistence_preproc[0, :, :, 0] - output_nn_all_norm[0, :, :, 0]]
                #
                # viz_obj.mincbar = -0.5
                # viz_obj.maxcbar = 0.5
                # viz_obj.plot_2d_data_np(fields, var_names=titles, title=F'{output_fields} {year}_{day_of_year:03d}',
                #                         file_name_prefix=F'{year}_error_{day_of_year:03d}', plot_mode=PlotMode.RASTER,
                #                        flip_data=True)


                # if (id_file % 20 == 0):
                #     errors_df.plot(title='CNN Comparison')
                #     plt.show()

if __name__ == '__main__':
    main()
