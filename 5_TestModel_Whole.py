# %%
import os
from os import listdir
import sys
sys.path.append("eoas_pyutils")
sys.path.append("eoas_pyutils/hycom_utils/python")

import matplotlib.pyplot as plt
from scipy.ndimage import convolve, zoom, median_filter, maximum_filter, gaussian_filter, minimum_filter, percentile_filter, spline_filter
from io_project.read_utils import generateXandY2D, normalizeData

import tensorflow as tf
from tensorflow.keras.utils import plot_model
from io_utils.io_netcdf import read_netcdf, read_netcdf_xr
from io_utils.io_common import create_folder
from os.path import join
import numpy as np
import pandas as pd
import time

from config.MainConfig_2D import get_prediction_params
from constants_proj.AI_proj_params import PredictionParams, ProjTrainingParams, PreprocParams
from config.PreprocConfig import get_preproc_config
from ai_common.models.modelSelector import select_2d_model
from models_proj.models import *
from ai_common.constants.AI_params import TrainingParams, ModelParams
from datetime import datetime, timedelta

from sklearn.metrics import mean_squared_error

from ExtraUtils.NamesManipulation import *
from viz_utils.eoa_viz import EOAImageVisualizer, BackgroundType
from viz_utils.eoa_viz import select_colormap

from hycom.io import read_hycom_fields, read_hycom_coords, read_field_names


def test_model(config, years):
    input_folder = config[PredictionParams.input_folder]
    output_folder = config[PredictionParams.output_folder]
    output_fields = config[ProjTrainingParams.output_fields]
    model_weights_file = config[PredictionParams.model_weights_file]
    output_imgs_folder = config[PredictionParams.output_imgs_folder]
    field_names = config[ProjTrainingParams.fields_names]
    comp_field_names = config[ProjTrainingParams.fields_names_composite]
    obs_field_names = config[ProjTrainingParams.fields_names_obs]
    rows = config[ProjTrainingParams.rows]
    cols = config[ProjTrainingParams.cols]
    run_name = config[TrainingParams.config_name]
    norm_type = config[ProjTrainingParams.norm_type]
    preproc_config = get_preproc_config()

    input_folder_background = preproc_config[PreprocParams.input_folder_hycom]
    input_folder_increment = preproc_config[PreprocParams.input_folder_tsis]
    input_folder_observations = preproc_config[PreprocParams.input_folder_obs]

    output_imgs_folder = join(output_imgs_folder, run_name)
    create_folder(output_imgs_folder)

    # *********** Chooses the proper model ***********
    print('Reading model ....')

    net_type = config[ProjTrainingParams.network_type]
    if net_type == NetworkTypes.UNET or net_type == NetworkTypes.UNET_MultiStream:
        model = select_2d_model(config, last_activation=None)
    if net_type == NetworkTypes.SimpleCNN_2:
        model = simpleCNN(config, nn_type="2d", hid_lay=2, out_lay=2)
    if net_type == NetworkTypes.SimpleCNN_4:
        model = simpleCNN(config, nn_type="2d", hid_lay=4, out_lay=2)
    if net_type == NetworkTypes.SimpleCNN_8:
        model = simpleCNN(config, nn_type="2d", hid_lay=8, out_lay=2)
    if net_type == NetworkTypes.SimpleCNN_16:
        model = simpleCNN(config, nn_type="2d", hid_lay=16, out_lay=2)

    plot_model(model, to_file=join(output_folder, F'running.png'), show_shapes=True)

    # *********** Reads the weights***********
    print('Reading weights ....')
    model.load_weights(model_weights_file)

    # *********** Read files to predict***********
    increment_files = np.array([join(input_folder_increment, x).replace(".a", "") for x in os.listdir(input_folder_increment) if x.endswith('.a')])
    increment_files.sort()

    z_layers = [0]
    var_file = join(input_folder, "cov_mat", "tops_ias_std.nc")
    field_names_std = config[ProjTrainingParams.fields_names_var]
    if len(field_names_std) > 0:
        input_fields_std = read_netcdf(var_file, field_names_std, z_layers)
    else:
        input_fields_std = []

    # Selects the min max color values
    cminmax_out = getMinMaxCbar([F"{x}_out" for x in output_fields])
    cminmax_model = getMinMaxCbar(field_names)
    cminmax_obs = getMinMaxCbar(obs_field_names)
    cminmax_comp = getMinMaxCbar(comp_field_names)
    cminmax_std = getMinMaxCbar(field_names_std)
    cminmax_error = getMinMaxCbar([F"error_{x}" for x in output_fields])

    # Selects the colormap to use for each field
    cmap_out =   [select_colormap(x) for x in output_fields]
    cmap_model = [select_colormap(x) for x in field_names]
    cmap_comp =  [select_colormap(x) for x in comp_field_names]
    cmap_obs =   [select_colormap(x) for x in obs_field_names]
    cmap_std =   [select_colormap(x) for x in field_names_std]
    cmap_error = [F"error{select_colormap(x)}" for x in output_fields]

    # Selects the colormap label to use for each field
    cmap_label_out = getFieldUnits(output_fields)
    cmap_label_model = getFieldUnits(field_names)
    cmap_label_comp = getFieldUnits(comp_field_names)
    cmap_label_obs = getFieldUnits(obs_field_names)
    cmap_label_std = getFieldUnits(field_names_std)
    cmap_label_error = getFieldUnits([F"error_{x}" for x in output_fields])

    all_whole_mean_times = []
    all_whole_sum_times = []
    all_whole_rmse = []

    tot_rows = 384
    tot_cols = 525

    coords_file = "/unity/g1/abozec/TSIS/GOMb0.04/topo/regional.grid.a"
    print(F"The coords available are: {read_field_names(coords_file)}")
    coords = read_hycom_coords(coords_file, ['plon:', 'plat:'])
    lons = coords['plon']
    lats = coords['plat']
    print("Done!")

    test_files = [x for x in increment_files if any(x.find(str(y)) != -1 for y in years)]

    # If one of the years is 2002, then we start from the 576 index, if not from 0
    if 2002 in years:
        start_test_idx = 0
    else:
        start_test_idx = 576+73

    test_files = test_files[start_test_idx:]
    # All files
    successful_files = []
    successful_dates = []

    # Only to simulate multiple z-layers as a batch
    # sim_z_layers = 41
    # new_x = np.zeros((sim_z_layers, 384, 520, 7), dtype=np.float16)



    for id_file, c_file in enumerate(test_files):
        # Find current and next date
        sp_name = c_file.split("/")[-1].split(".")[1]
        c_datetime = datetime.strptime(sp_name, "%Y_%j_18")
        c_datetime_next_day = c_datetime + timedelta(days=1)
        day_of_year = c_datetime.timetuple().tm_yday
        c_day_str = c_datetime.strftime("%Y-%m-%d")
        print(F"=================== Day of year {c_day_str}_{day_of_year} ==========================")

        model_file_name = join(input_folder_background, F"022_archv.{c_datetime.strftime('%Y_%j')}_18.a")
        # model_file_name = join(input_folder_background, F"020_archv.{c_datetime.strftime('%Y_%j')}_18.a")
        increment_file_name = c_file
        obs_file_name = join(input_folder_observations, F"tsis_obs_gomb4_{c_datetime_next_day.strftime('%Y%m%d')}00.nc")

        # *********************** Reading files **************************
        try:
            input_fields_model = read_hycom_fields(model_file_name, field_names, z_layers)

            input_fields_obs = read_netcdf_xr(obs_file_name, obs_field_names, z_layers)
            output_field_increment = read_hycom_fields(increment_file_name, output_fields, z_layers)
        except Exception as e:
            print(F"Couldn't find all files for date {c_day_str}: {input_folder_background}, {input_folder_increment}")
            continue
        # ******************* Normalizing and Cropping Data *******************
        this_file_times = []

        try:
            perc_ocean = 0
            input_data, y_data = generateXandY2D(input_fields_model, input_fields_obs, input_fields_std, output_field_increment,
                                                 field_names+comp_field_names, obs_field_names, field_names_std, output_fields,
                                                 0, 0, tot_rows, tot_cols, norm_type=norm_type, perc_ocean=perc_ocean)
                                                # start_row, start_col, rows, cols, norm_type=norm_type, perc_ocean=perc_ocean)
        except Exception as e:
            print(F"Exception {e}")

        # ******************* Replacing nan values *********
        # We set a value of 0.5 on the land. Trying a new loss function that do not takes into account land
        input_data_nans = np.isnan(input_data)
        input_data = np.nan_to_num(input_data, nan=0)
        y_data = np.nan_to_num(y_data, nan=-0.5)
        cnn_output = np.zeros(y_data.shape)

        # Make predictions of all the domain
        for c_row in range(0, tot_rows, rows):
            for c_col in range(0, tot_cols, cols):
                # This part fills the whole domain, but most of the time the last row and cols are computed twice
                if c_row + rows >= tot_rows:
                    s_row = tot_rows-rows
                else:
                    s_row = c_row
                if c_col + cols >= tot_cols:
                    s_col = tot_cols-cols
                else:
                    s_col = c_col

                print(F"{s_row}:{s_row+rows}, {s_col}:{s_col+cols}")
                X = np.expand_dims(input_data[s_row:s_row+rows, s_col:s_col+cols,:], axis=0)
                Y = np.expand_dims(y_data[s_row:s_row+rows, s_col:s_col+cols,:], axis=0)
                X = X.astype(np.float16)
                Y = Y.astype(np.float16)

                #=====================  Make the prediction of the network =======================
                # --------------- DELETE THIS JUST FOR TESTING TIME TO PREDICT ALL THE 30 LEVELS----------------
                # for i in range(sim_z_layers):
                #     new_x[i,:,:,:] = X

                # tensor = tf.convert_to_tensor(new_x)
                # # Place the tensor on the GPU
                # with tf.device('/GPU:0'):
                #     gpu_tensor = tf.identity(tensor)

                # start = time.time()
                # output_nn_original_multiple = model.predict(gpu_tensor)
                # toc = time.time() - start
                # # Print current time: 
                # print(F"Time to predict all the {sim_z_layers} levels: {toc:0.4f} seconds")

                # this_file_times.append(toc)

                # if id_file == 0:  # REMOVE THIS IF FOR NORMAL RUN
                #     output_nn_original = model.predict(X)
                # --------------- DELETE THIS JUST FOR TESTING TIME TO PREDICT ALL THE 30 LEVELS----------------

                # --------------- Normal prediction ---------------
                start = time.time()
                output_nn_original = model.predict(X)
                toc = time.time() - start
                this_file_times.append(toc)
                # --------------- Normal prediction ---------------

                # Make nan all values inside the land
                land_indexes = Y == -0.5
                output_nn_original[land_indexes] = np.nan

                cnn_output[s_row:s_row+rows, s_col:s_col+cols] = output_nn_original[0,:,:,:]

        # ==== Denormalizing all input and outputs
        denorm_cnn = denormalizeData(cnn_output, output_fields, PreprocParams.type_inc, norm_type)
        denorm_y = denormalizeData(y_data, output_fields, PreprocParams.type_inc, norm_type)
        input_types = [PreprocParams.type_model for i in field_names+comp_field_names] + [PreprocParams.type_obs for i in obs_field_names] + [PreprocParams.type_std for i in field_names_std]
        denorm_input = denormalizeData(input_data, field_names+comp_field_names+obs_field_names+field_names_std, input_types, norm_type)

        # Recover the original land areas, they are lost after denormalization
        land_indexes = y_data == -0.5
        denorm_y[land_indexes] = np.nan

        # Adding back mask to all the input variables
        denorm_input[input_data_nans] = np.nan

        error = denorm_y - denorm_cnn
        no_zero_ids = np.count_nonzero(np.logical_not(np.isnan(cnn_output)))

        rmse_cnn = np.sqrt( np.nansum( (denorm_y - denorm_cnn)**2 , axis=(0,1))/no_zero_ids)
        mae_cnn = np.nansum( np.abs(denorm_y - denorm_cnn), axis=(0,1))/no_zero_ids

        # all_whole_rmse.append(rmse_cnn.value)
        all_whole_rmse.append(rmse_cnn.item())
        all_whole_mean_times.append(np.mean(np.array(this_file_times)))
        all_whole_sum_times.append(np.sum(np.array(this_file_times)))

        if id_file % 10 == 0: # Plot 10% of the times
            all_cmin = cminmax_model[0]+cminmax_comp[0]+cminmax_obs[0]+cminmax_std[0]+cminmax_out[0]+cminmax_out[0]+cminmax_error[0]
            all_cmax = cminmax_model[1]+cminmax_comp[1]+cminmax_obs[1]+cminmax_std[1]+cminmax_out[1]+cminmax_out[1]+cminmax_error[1]

            size = 2
            filter = 1/(2**2) * np.ones((size,size))

            eps = .001
            # ------------------ Smoothing fields for visualization -------------
            obs_ssh_idx = len(field_names) + len(comp_field_names)
            ssh_diff_idx = len(field_names)
            model_ssh_idx = 0 # TODO review this is the SSH index for the model

            # Smooths SSH (Assumes the first obs field is always SSH)
            temp_field = denorm_input[:,:,obs_ssh_idx]  # Selects SSH
            temp_field = np.nan_to_num(temp_field, 0)
            temp_field = gaussian_filter(temp_field, 1)
            temp_field[np.logical_and(temp_field >= -eps, temp_field <= eps)] = np.nan
            denorm_input[:,:,obs_ssh_idx] = temp_field

            # # Smooths DIFF_SSH (assumes it is the first composite field)
            temp_field = denorm_input[:,:,ssh_diff_idx] # Reads the "DIFF" field
            temp_field = np.nan_to_num(temp_field, 0)
            temp_field = gaussian_filter(temp_field, 1)
            # temp_field[temp_field == 0] = np.nan
            temp_field[np.logical_and(temp_field >= -eps, temp_field <= eps)] = np.nan
            temp2 = denorm_y[:, :, model_ssh_idx].copy()
            temp_idxs = np.logical_not(np.isnan(temp_field))
            temp2[temp_idxs] = temp_field[temp_idxs]
            # denorm_input[:,:,ssh_diff_idx] = temp2 # Reassigned the input diff
            denorm_input[:,:,ssh_diff_idx] = temp_field# Only smoothed diff

            # Smooths Observations
            temp_field = denorm_input[:,:,len(field_names) + 1] # Reads the "DIFF" field
            temp_field = np.nan_to_num(temp_field, 0)
            temp_field = convolve(temp_field, filter)  # Here is the convolution (extension of values)
            temp_field[temp_field == 0] = np.nan
            denorm_input[:,:,len(field_names) + 1] = temp_field  # Reassigned

            rmse_txts =[F"{rmse_cnn[i]:0.4f}" for i,x in enumerate(output_fields)]

            # # ================== Displays ALL ================
            viz_obj = EOAImageVisualizer(output_folder=output_imgs_folder, disp_images=False,
                                         lats=lats, lons=lons,
                                         max_imgs_per_row=5,
                                         show_var_names=True )
            viz_obj.plot_2d_data_np(np.concatenate((denorm_input.swapaxes(0,2), denorm_y.swapaxes(0,2), denorm_cnn.swapaxes(0,2), error.swapaxes(0,2))),
                                        var_names=[F"in_model_{x}" for x in field_names] +
                                                  [F"in_comp_{x}" for x in comp_field_names] +
                                                  [F"in_obs_{x}" for x in obs_field_names] +
                                                  [F"out_inc_{x} (MAE {np.nanmean(np.abs(denorm_y[:,:,i])):0.2f})" for i,x in enumerate(output_fields)] +
                                                  [F"cnn_{x}" for x in output_fields] +
                                                  [F"Difference RMSE {rmse_cnn[i]:0.4f} MAE {mae_cnn[i]:0.4f}" for i, x in enumerate(output_fields)],
                                        file_name_prefix=F"Global_Input_and_CNN_{sp_name}",
                                        rot_90=True,
                                        flip_data=True,
                                        # cmap=cmap_model+cmap_comp+cmap_obs+cmap_std+cmap_out+cmap_out+cmap_error,
                                        # cmap_labels=cmap_label_model+cmap_label_comp+cmap_label_obs+cmap_label_out+cmap_label_out+cmap_label_error,
                                        # cols_per_row=len(field_names),
                                        # title=F"Input data: {field_names} and obs {obs_field_names}, increment {output_fields}, cnn {output_fields}")
                                        mincbar=all_cmin,maxcbar=all_cmax,
                                        title=F"RMSE {rmse_txts} m {sp_name}")

            # # ================== Displays only CNN and TSIS with RMSE ================
            viz_obj = EOAImageVisualizer(output_folder=output_imgs_folder, disp_images=False,
                                         max_imgs_per_row=3,
                                         lats=lats, lons=lons,
                                        background=BackgroundType.WHITE,
                                        show_var_names = True )
            viz_obj.plot_2d_data_np(np.concatenate((denorm_y.swapaxes(0,2), denorm_cnn.swapaxes(0,2), error.swapaxes(0,2))),
                                        var_names=[F"TSIS {x}" for x in output_fields] + [F"CNN {x}" for x in output_fields] + [F'TSIS - CNN srfhgt \n (Mean RMSE {rmse_cnn[i]:0.4f} C)' for i in range(len(output_fields))],
                                        file_name_prefix=F"Global_WholeOutput_CNN_TSIS_{sp_name}",
                                        rot_90=True,
                                        flip_data=True,
                                        # cmap=cmap_out+cmap_out+cmap_error,
                                        mincbar=cminmax_out[0] + cminmax_out[0] + cminmax_error[0],
                                        maxcbar=cminmax_out[1] + cminmax_out[1] + cminmax_error[1],
                                        # cmap_labels=cmap_label_out+cmap_label_out+cmap_label_error,
                                        title=F"RMSE {rmse_txts} m {sp_name}")

        successful_dates.append(c_day_str)
        successful_files.append(str(c_file))

    print("DONE ALL FILES!!!!!!!!!!!!!")
    dic_summary = {
        "File": successful_files,
        "rmse": all_whole_rmse,
        "dates": successful_dates,
        "times mean": all_whole_mean_times,
        "times sum": all_whole_sum_times,
    }
    df = pd.DataFrame.from_dict(dic_summary)
    df.to_csv(join(output_imgs_folder, "Global_RMSE_and_times.csv"))


def getFieldUnits(fields):
    cmaps_fields = []
    degree_sign = u"\N{DEGREE SIGN}"
    for c_field in fields:
        if c_field == "srfhgt" or c_field == "ssh":
            cmaps_fields.append("meters")
        elif c_field == "temp" or c_field == "sst" or c_field == "temp":
            cmaps_fields.append(F"{degree_sign}C")
        elif c_field == "salin" or c_field == "sss" or c_field == "sal":
            cmaps_fields.append("?")
        elif c_field == "u-vel.":
            cmaps_fields.append("m/s")
        elif c_field == "v-vel.":
            cmaps_fields.append("m/s")
        elif c_field == "error":
            cmaps_fields.append("")
        else:
            cmaps_fields.append("")
    return cmaps_fields

def getMinMaxCbar(fields):
    mincbar = []
    maxcbar = []
    for c_field in fields:
        if c_field == "srfhgt" or c_field == "ssh":
            maxcbar.append(.4)
            mincbar.append(-.4)
        elif c_field == "temp" or c_field == "sst":
            maxcbar.append(np.nan)
            mincbar.append(np.nan)
        elif c_field == "srfhgt_out":
            maxcbar.append(.4)
            mincbar.append(-.4)
        elif c_field == "temp_out" :
            maxcbar.append(2.)
            mincbar.append(-2.)
        elif c_field == "diff_ssh":
            maxcbar.append(0.1)
            mincbar.append(-0.1)
        elif c_field == "salin" or c_field == "sss" or c_field == "sal":
            maxcbar.append(np.nan)
            mincbar.append(np.nan)
        elif c_field == "u-vel.":
            maxcbar.append(np.nan)
            mincbar.append(np.nan)
        elif c_field == "v-vel.":
            maxcbar.append(np.nan)
            mincbar.append(np.nan)
        elif c_field == "error_srfhgt":
            maxcbar.append(0.2)
            mincbar.append(-0.2)
        elif c_field == "error_temp":
            maxcbar.append(2.0)
            mincbar.append(-2.0)
        else:
            maxcbar.append(np.nan)
            mincbar.append(np.nan)
    return mincbar, maxcbar


def denormalizeData(input, fields, data_type, norm_type):
    output = np.zeros(input.shape)
    for field_idx, c_field in enumerate(fields):
        # Denormalizing data...

        if c_field != "diff_sst" and c_field != "diff_ssh" and c_field != "topo":
            if len(output.shape) == 4:
                if type(data_type) is list:
                    output[:, :, :, field_idx] = normalizeData(input[:, :, :, field_idx], c_field, data_type=data_type[field_idx], norm_type= norm_type, normalize=False)
                else:
                    output[:, :, :, field_idx] = normalizeData(input[:, :, :, field_idx], c_field, data_type=data_type, norm_type= norm_type, normalize=False)
            elif len(output.shape) == 3:
                if type(data_type) is list:
                    output[:, :, field_idx] = normalizeData(input[:, :, field_idx], c_field, data_type=data_type[field_idx], norm_type= norm_type, normalize=False)
                else:
                    output[:, :, field_idx] = normalizeData(input[:, :, field_idx], c_field, data_type=data_type, norm_type= norm_type, normalize=False)
            else:
                print("ERROR Dimensions not found in denormalization!!!")
                exit()
        else:
            output[:, :, field_idx] = input[:, :, field_idx]

    return output


def verifyBoundaries(start_col, cols, tot_cols):
    donecol = False
    if start_col + cols < tot_cols - 1:
        start_col += cols
    elif start_col + cols > tot_cols-1:
        start_col = tot_cols - cols - 1
    elif (start_col + cols) == tot_cols-1:
        donecol = True
    return start_col, donecol

if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    config = get_prediction_params()
    years = [2009, 2010]  # Which years to evaluate the model
    # years = [2002, 2006]  # Which years to evaluate the model
    # -------- For single model testing (not easy to test because everything must be defined in MainConfig_2D.py)--------------
    # print("Testing single model....")
    # test_model(config)

    # -------- For all summary model testing --------------
    print("Testing all the models inside summary.csv ....")
    # summary_file = "/data/HYCOM/DA_HYCOM_TSIS/SUMMARY/Summary_Only_Best.csv"
    # summary_file = "/unity/f1/ozavala/DATA/NN_HYCOM_TSIS/SUMMARY/Summary_Only_Best.csv"
    summary_file = "/unity/f1/ozavala/OUTPUTS/DA_HYCOM_TSIS/TrainingSkynetPaperReviews/SUMMARY/summary.csv"
    df = pd.read_csv(summary_file)

    for model_id in range(len(df)):
        model = df.iloc[model_id]
        # Setting Network type (only when network type is UNET)
        name = model["Name"]
        print(F"Model id: {model_id}, name: {name}")
        network_arch, network_type = getNeworkArchitectureAndTypeFromName(getNetworkTypeTxt(name))
        config[ModelParams.MODEL] = network_arch
        config[ProjTrainingParams.network_type] = network_type
        # Setting input vars
        inputs_path = model["Path"].replace("models", "Parameters")
        input_file = join(inputs_path, listdir(inputs_path)[0]) # This should give you the file name with the inputs
        in_df = pd.read_csv(input_file)
        # Setting model vars
        # model_fields = in_df["Model"].item().split(',')  # It got modified at some point
        model_fields = in_df["Model_input"].item().split(',')
        config[ProjTrainingParams.fields_names] = model_fields
        # Setting obs vars
        obs_fields = in_df["Obs"].item().split(',')
        config[ProjTrainingParams.fields_names_obs] = obs_fields
        # Setting composite vars
        comp_fields = in_df["Comp"].item().split(',')
        config[ProjTrainingParams.fields_names_composite] = comp_fields
        # Setting output vars
        output_fields = in_df["output"].item().split(',')
        config[ProjTrainingParams.output_fields] = output_fields
        config[ModelParams.OUTPUT_SIZE] = len(config[ProjTrainingParams.output_fields])
        print(F"Input fields: {model_fields}, {obs_fields}, {comp_fields}  Output fields: {output_fields}")
        # Model parameters
        # filter_size = int(in_df["Model_params"].item().split(':')[1]) # It also got modified at some point
        filter_size = int(in_df["filter_size"])
        config[ModelParams.FILTER_SIZE] = filter_size

        # Setting BBOX
        grows, gcols, bboxtxt = getBBOXandText(name)
        config[ModelParams.INPUT_SIZE][0] = grows
        config[ModelParams.INPUT_SIZE][1] = gcols
        config[ModelParams.INPUT_SIZE][2] = len(model_fields) + len(obs_fields) + len(comp_fields) + 2 # +2 for lat and lon
        config[ProjTrainingParams.rows] = grows
        config[ProjTrainingParams.cols] = gcols
        # Setting model weights file
        config[PredictionParams.model_weights_file] = join(model["Path"],model["Name"])
        print(F"Model's weight file: {config[PredictionParams.model_weights_file]}")
        # Set the name of the network
        run_name = model["Path"].replace("/models","").split("/")[-1]  # The runname
        config[TrainingParams.config_name] = run_name
        test_model(config, years)
# %%
