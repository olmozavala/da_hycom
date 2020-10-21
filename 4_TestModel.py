import os
from io_project.read_utils import generateXandY, normalizeData

from numpy.distutils.system_info import flame_info
from tensorflow.keras.utils import plot_model
from inout.io_netcdf import read_netcdf
from os.path import join, exists
import numpy as np
import numpy.ma as ma
import pandas as pd

from img_viz.eoa_viz import EOAImageVisualizer
from config.MainConfig import get_prediction_params
from constants_proj.AI_proj_params import PredictionParams, ProjTrainingParams, PreprocParams
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
    norm_type = config[ProjTrainingParams.norm_type]

    output_imgs_folder = join(output_imgs_folder, run_name)

    # *********** Chooses the proper model ***********
    print('Reading model ....')
    model = select_2d_model(config, last_activation='relu')
    # model = simpleCNN(config)
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
        for start_row in np.arange(0, 891-rows, rows):
            for start_col in np.arange(0, 1401-cols, cols):
        # for start_row in np.arange(0, 891-rows, 200):
        #     for start_col in np.arange(0, 1401-cols, 100):
                # Generate the proper inputs for the NN
                try:
                    input_data, y_data = generateXandY(input_fields_model, input_fields_obs, output_field_increment, field_names, obs_field_names, output_fields,
                                                       start_row, start_col, rows, cols, norm_type=norm_type)
                except Exception as e:
                    print(F"Failed for {c_file} row:{start_row} col:{start_col}")
                    continue

                X = np.expand_dims(input_data, axis=0)
                Y = np.expand_dims(y_data, axis=0)

                # ******************* Replacing nan values *********
                # We set a value of 0.5 on the land. Trying a new loss function that do not takes into account land
                X = np.nan_to_num(X, nan=0)
                Y = np.nan_to_num(Y, nan=-0.5)

                # Make the prediction of the network
                output_nn_original = model.predict(X, verbose=1)

                # Make nan all values inside the land
                land_indexes = Y == -0.5
                output_nn_original[land_indexes] = np.nan

                # Denormalize the data to the proper units in each field
                denorm_cnn_output = np.zeros(output_nn_original.shape)
                denorm_y = np.zeros(Y.shape)
                for field_idx, c_field in enumerate(output_fields):
                    denorm_cnn_output[:, :, :, field_idx] = normalizeData(output_nn_original[:, :, :, field_idx], c_field, data_type = PreprocParams.type_inc, norm_type= norm_type, normalize=False)
                    denorm_y[:, :, :, field_idx] = normalizeData(Y[:, :, :, field_idx], c_field, data_type = PreprocParams.type_inc, norm_type= norm_type, normalize=False)
                # Recover the original land areas, they are lost after denormalization
                denorm_y[land_indexes] = np.nan

                # Remove the 'extra dimension'
                denorm_cnn_output = np.squeeze(denorm_cnn_output)
                denorm_y = np.squeeze(denorm_y)
                if len(denorm_cnn_output.shape) == 2: # In this case we only had one output and we need to make it 'array' to plot
                    denorm_cnn_output = np.expand_dims(denorm_cnn_output, axis=2)
                    denorm_y = np.expand_dims(denorm_y, axis=2)

                # Compute RMSE
                rmse_cnn = np.zeros(len(output_fields))
                for i in range(len(output_fields)):
                    ocean_indexes = np.logical_not(np.isnan(denorm_y[:,:,i]))
                    rmse_cnn[i] = np.sqrt(mean_squared_error(denorm_cnn_output[:,:,i][ocean_indexes], denorm_y[:,:,i][ocean_indexes]))

                # viz_obj = EOAImageVisualizer(output_folder=output_imgs_folder, disp_images=False, mincbar=mincbar, maxcbar=maxcbar)
                viz_obj = EOAImageVisualizer(output_folder=output_imgs_folder, disp_images=False)

                # ================== This displays all inputs and the cnn output ================
                # viz_obj.plot_2d_data_np_raw(np.concatenate((X[0,:,:,:].swapaxes(0,2), Y[0,:,:,:].swapaxes(0,2))),
                viz_obj.plot_2d_data_np_raw(np.concatenate((input_data.swapaxes(0,2), denorm_y.swapaxes(0,2), denorm_cnn_output.swapaxes(0,2))),
                                            var_names=[F"in_model_{x}" for x in field_names] +
                                                      [F"in_obs_{x}" for x in obs_field_names] +
                                                      [F"out_inc_{x}" for x in output_fields] +
                                                      [F"cnn_{x}" for x in output_fields],
                                            file_name=F"Input_and_CNN_{c_file}_{start_row:03d}_{start_col:03d}",
                                            rot_90=True,
                                            cols_per_row=len(field_names),
                                            title=F"Input data: {field_names} and obs {obs_field_names}, increment {output_fields}, cnn {output_fields}")

                # =========== Making the same color bar for desired output and the NN =====================
                mincbar = [np.nanmin(denorm_y[:, :, x]) for x in range(denorm_cnn_output.shape[-1])]
                maxcbar = [np.nanmax(denorm_y[:, :, x]) for x in range(denorm_cnn_output.shape[-1])]
                error = (denorm_y - denorm_cnn_output).swapaxes(0,2)
                mincbarerror = [np.nanmin(error[i,:,:]) for i in range(len(output_fields))]
                maxcbarerror = [np.nanmax(error[i,:,:]) for i in range(len(output_fields))]
                viz_obj = EOAImageVisualizer(output_folder=output_imgs_folder, disp_images=False,
                                             mincbar=mincbar + mincbar + mincbarerror,
                                             maxcbar=maxcbar + maxcbar + maxcbarerror)

                # ================== Displays desired output ALL TOGETHER ================
                viz_obj.output_folder = join(output_imgs_folder,'JoinedErrrorCNN')
                viz_obj.plot_2d_data_np_raw(np.concatenate((denorm_cnn_output.swapaxes(0,2), denorm_y.swapaxes(0,2), error), axis=0),
                                             var_names=[F"CNN {x}" for x in output_fields] + [F"TSIS INC {x}" for x in output_fields] + [F'RMSE {c_rmse_cnn:0.4f}' for c_rmse_cnn in rmse_cnn],
                                            file_name=F"AllError_{c_file}_{start_row:03d}_{start_col:03d}",
                                            rot_90=True,
                                            cols_per_row=len(output_fields),
                                            title=F"{output_fields} RMSE: {np.mean(rmse_cnn):0.5f}")

                # ====================================================================
                # ================= Individual figures for error, cnn and expected ================
                # ====================================================================
                #
                # # ================== Displays CNN error ================
                # viz_obj.output_folder = join(output_imgs_folder,'error')
                # viz_obj.plot_2d_data_np_raw((denorm_y - denorm_cnn_output).swapaxes(0,2),
                #                             var_names= [F"out_inc_{x}" for x in output_fields],
                #                             file_name=F"ERROR_CNN_{c_file}_{start_row:03d}_{start_col:03d}",
                #                             rot_90=True,
                #                             title=F"RMSE {output_fields}  {np.mean(rmse_cnn):0.5f}")
                #
                # # =========== Making the same color bar for desired output and the NN =====================
                # # mincbar = [min(np.nanmin(denorm_y[:, :, x]), np.nanmin(denorm_cnn_output[:, :, x])) for x in range(denorm_cnn_output.shape[-1])]
                # # maxcbar = [max(np.nanmax(denorm_y[:, :, x]), np.nanmax(denorm_cnn_output[:, :, x])) for x in range(denorm_cnn_output.shape[-1])]
                # mincbar = [np.nanmin(denorm_y[:, :, x]) for x in range(denorm_cnn_output.shape[-1])]
                # maxcbar = [np.nanmax(denorm_y[:, :, x]) for x in range(denorm_cnn_output.shape[-1])]
                # viz_obj = EOAImageVisualizer(output_folder=output_imgs_folder, disp_images=False, mincbar=mincbar, maxcbar=maxcbar)
                #
                # # ================== Displays desired output ================
                # viz_obj.output_folder = join(output_imgs_folder,'expected')
                # viz_obj.plot_2d_data_np_raw(denorm_y.swapaxes(0,2),
                #                             var_names=[F"out_inc_{x}" for x in output_fields],
                #                             file_name=F"Expected_output_{c_file}_{start_row:03d}_{start_col:03d}",
                #                             rot_90=True,
                #                             title=F"Expected {output_fields}")
                #
                # # ================== Displays only CNN output ================
                # viz_obj.output_folder = join(output_imgs_folder,'cnn')
                # viz_obj.plot_2d_data_np_raw(denorm_cnn_output.swapaxes(0,2),
                #                             var_names= [F"out_inc_{x}" for x in output_fields],
                #                             file_name=F"Only_CNN_{c_file}_{start_row:03d}_{start_col:03d}",
                #                             rot_90=True,
                #                             title=F"CNN {output_fields}")



if __name__ == '__main__':
    main()

##

