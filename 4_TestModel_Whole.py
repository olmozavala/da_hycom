import os
from io_project.read_utils import generateXandY2D, normalizeData

from tensorflow.keras.utils import plot_model
from inout.io_netcdf import read_netcdf, read_netcdf_xr
from os.path import join
import numpy as np
import pandas as pd
import time
import cmocean

from img_viz.eoa_viz import EOAImageVisualizer
from config.MainConfig_2D import get_prediction_params
from constants_proj.AI_proj_params import PredictionParams, ProjTrainingParams, PreprocParams
from config.PreprocConfig import get_preproc_config
from models.modelSelector import select_2d_model
from models_proj.models import *
from constants.AI_params import TrainingParams, ModelParams
from img_viz.common import create_folder
from datetime import datetime
from inout.io_hycom import read_hycom_fields

from sklearn.metrics import mean_squared_error

from ExtraUtils.NamesManipulation import *
from eoas_utils.VizUtilsProj import chooseCMAP, getMinMaxPlot

def main():

    config = get_prediction_params()
    # -------- For single model testing --------------
    # print("Testing single model....")
    # test_model(config)

    # -------- For all summary model testing --------------
    print("Testing all the models inside summary.csv ....")
    summary_file = "/data/HYCOM/DA_HYCOM_TSIS/SUMMARY/summary.csv"
    df = pd.read_csv(summary_file)

    for model_id in range(len(df)):
        model = df.iloc[model_id]
        # Setting Network type (only when network type is UNET)
        name = model["Name"]
        network_arch, network_type = getNeworkArchitectureAndTypeFromName(getNetworkTypeTxt(name))
        config[ModelParams.MODEL] = network_arch
        config[ProjTrainingParams.network_type] = network_type
        # Setting input vars
        # model_fields, obs_fields, var_fields = getInputFields(name)
        var_fields = getInputVarFields(name)
        model_fields = getInputFields(name)
        config[ProjTrainingParams.fields_names] = model_fields
        obs_fields = getObsFieldsTxt(name)
        config[ProjTrainingParams.fields_names_obs] = obs_fields
        comp_fields = getCompostieFields(name)
        config[ProjTrainingParams.fields_names_composite] = comp_fields
        print(F"Input fields: {model_fields}, {obs_fields}, {var_fields}, {comp_fields}")
        # config[ProjTrainingParams.fields_names] = model_fields
        # config[ProjTrainingParams.fields_names_obs] = obs_fields
        config[ProjTrainingParams.fields_names_var] = var_fields
        # Setting BBOX
        grows, gcols, bboxtxt = getBBOXandText(name)
        config[ModelParams.INPUT_SIZE][0] = grows
        config[ModelParams.INPUT_SIZE][1] = gcols
        config[ModelParams.INPUT_SIZE][2] = len(model_fields) + len(obs_fields) + len(var_fields) + len(comp_fields)
        config[ProjTrainingParams.rows] = grows
        config[ProjTrainingParams.cols] = gcols
        # Setting output vars
        # output_fields = getOutputFields(name)
        # config[ProjTrainingParams.output_fields] = output_fields
        config[ProjTrainingParams.output_fields] = ['srfhgt']
        # Setting model weights file
        config[PredictionParams.model_weights_file] = join(model["Path"],model["Name"])
        print(F"Model's weight file: {config[PredictionParams.model_weights_file]}")
        # Set the name of the network
        run_name = model["Path"].replace("/models","").split("/")[-1]  # The runname
        config[TrainingParams.config_name] = run_name
        test_model(config)

def test_model(config):
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

    input_folder_background =   preproc_config[PreprocParams.input_folder_hycom]
    input_folder_increment =    preproc_config[PreprocParams.input_folder_tsis]
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
    cminmax_out = getMinMaxCbar(output_fields)
    # cminmax_model = getMinMaxCbar(field_names)
    cminmax_model = getMinMaxCbar(["" for x in field_names])
    cminmax_obs = getMinMaxCbar(obs_field_names)
    cminmax_comp = getMinMaxCbar(comp_field_names)
    cminmax_std = getMinMaxCbar(field_names_std)
    cminmax_error = getMinMaxCbar(["error"])

    # Selects the colormap to use for each field
    cmap_out = chooseCMAP(output_fields)
    cmap_model = chooseCMAP(field_names)
    cmap_comp = chooseCMAP(comp_field_names)
    cmap_obs = chooseCMAP(obs_field_names)
    cmap_std = chooseCMAP(field_names_std)
    cmap_error = chooseCMAP(["error"])

    # Selects the colormap label to use for each field
    cmap_label_out = getFieldUnits(output_fields)
    cmap_label_model = getFieldUnits(field_names)
    cmap_label_comp = getFieldUnits(comp_field_names)
    cmap_label_obs = getFieldUnits(obs_field_names)
    cmap_label_std = getFieldUnits(field_names_std)
    cmap_label_error = getFieldUnits(["error"])

    all_whole_mean_times = []
    all_whole_sum_times = []
    all_whole_rmse = []

    tot_rows = 384
    tot_cols = 525

    start_test_idx = 583+73

    for id_file, c_file in enumerate(increment_files[start_test_idx:]):
        # Find current and next date
        sp_name = c_file.split("/")[-1].split(".")[1]
        c_datetime = datetime.strptime(sp_name, "%Y_%j_18")
        day_of_year = c_datetime.timetuple().tm_yday
        print(F"=================== Day of year {day_of_year} ==========================")

        model_file_name = join(input_folder_background, F"020_archv.{c_datetime.strftime('%Y_%j')}_18.a")
        increment_file_name = c_file
        obs_file_name = join(input_folder_observations, F"tsis_obs_gomb4_{c_datetime.strftime('%Y%m%d')}00.nc")

        # *********************** Reading files **************************
        input_fields_model = read_hycom_fields(model_file_name, field_names, z_layers)
        input_fields_obs = read_netcdf_xr(obs_file_name, obs_field_names, z_layers)
        output_field_increment = read_hycom_fields(increment_file_name, output_fields, z_layers)

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

                #=====================  Make the prediction of the network =======================
                start = time.time()
                output_nn_original = model.predict(X, verbose=1)
                toc = time.time() - start
                this_file_times.append(toc)

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

        rmse_cnn = np.sqrt( np.nansum( (denorm_y - denorm_cnn)**2 )/no_zero_ids)
        mae_cnn = np.nansum( np.abs(denorm_y - denorm_cnn))/no_zero_ids

        all_whole_rmse.append(rmse_cnn)
        all_whole_mean_times.append(np.mean(np.array(this_file_times)))
        all_whole_sum_times.append(np.sum(np.array(this_file_times)))

        # if day_of_year%30 == 0: # Plot 10% of the times
        if True: # Plot 10% of the times
            # viz_obj = EOAImageVisualizer(output_folder=output_imgs_folder, disp_images=False, mincbar=mincbar, maxcbar=maxcbar)
            viz_obj = EOAImageVisualizer(output_folder=output_imgs_folder, disp_images=False,
                                         mincbar=cminmax_model[0]+cminmax_comp[0]+cminmax_obs[0]+cminmax_std[0]+cminmax_out[0]+cminmax_out[0]+cminmax_error[0],
                                         maxcbar=cminmax_model[1]+cminmax_comp[1]+cminmax_obs[1]+cminmax_std[1]+cminmax_out[1]+cminmax_out[1]+cminmax_error[1])

            # viz_obj.plot_2d_data_np_raw(np.concatenate((input_data.swapaxes(0,2), Y[0,:,:,:].swapaxes(0,2), output_nn_original[0,:,:,:].swapaxes(0,2))),
            viz_obj.plot_2d_data_np_raw(np.concatenate((denorm_input.swapaxes(0,2), denorm_y.swapaxes(0,2), denorm_cnn.swapaxes(0,2), error.swapaxes(0,2))),
                                        var_names=[F"in_model_{x}" for x in field_names] +
                                                  [F"in_comp_{x}" for x in comp_field_names] +
                                                  [F"in_obs_{x}" for x in obs_field_names] +
                                                  [F"in_var_{x}" for x in field_names_std] +
                                                  [F"out_inc_{x} MAE ({np.nanmean(np.abs(denorm_y[:,:,i])):0.2f})" for i,x in enumerate(output_fields)] +
                                                  [F"cnn_{x}" for x in output_fields] +
                                                  [F"Difference RMSE {rmse_cnn:0.4f} MAE {mae_cnn:0.4f}"],
                                        file_name=F"Global_Input_and_CNN_{sp_name}",
                                        rot_90=True,
                                        cmap=cmap_model+cmap_comp+cmap_obs+cmap_std+cmap_out+cmap_out+cmap_error,
                                        cmap_labels=cmap_label_model+cmap_label_comp+cmap_label_obs+cmap_label_std+cmap_label_out+cmap_label_out+cmap_label_error,
                                        # cols_per_row=len(field_names),
                                        cols_per_row=4,
                                        # title=F"Input data: {field_names} and obs {obs_field_names}, increment {output_fields}, cnn {output_fields}")
                                        title=F"RMSE {rmse_cnn:0.4f} m {sp_name}")

            # # ================== Displays CNN and TSIS with RMSE ================
            # minmax = getMinMaxPlot(output_fields)[0]
            # viz_obj = EOAImageVisualizer(output_folder=output_imgs_folder, disp_images=False,
            #                              # mincbar=mincbar + mincbar + mincbarerror,
            #                              # maxcbar=maxcbar + maxcbar + maxcbarerror)
            #                              # mincbar=[minmax[0], minmax[0], max(minmax[0],-1)],
            #                              # maxcbar=[minmax[1], minmax[1], min(minmax[1],1)])
            #                             mincbar=[minmax[0], minmax[0], -1],
            #                             maxcbar=[minmax[1], minmax[1], 1])
            #
            # error_cmap = cmocean.cm.diff
            # viz_obj.output_folder = join(output_imgs_folder,'WholeOutput_CNN_TSIS')
            # viz_obj.plot_2d_data_np_raw([np.flip(cnn_output, axis=0), np.flip(y_data, axis=0), np.flip(error, axis=0)],
            #                             # var_names=[F"CNN INC {x}" for x in output_fields] + [F"TSIS INC {x}" for x in output_fields] + [F'TSIS - CNN (Mean RMSE {rmse_cnn:0.4f} m)'],
            #                             var_names=[F"CNN increment SSH" for x in output_fields] + [F"TSIS increment SSH" for x in output_fields] + [F'TSIS - CNN \n (Mean RMSE {rmse_cnn:0.4f} m)'],
            #                             file_name=F"Global_WholeOutput_CNN_TSIS_{c_file}",
            #                             rot_90=False,
            #                             cmap=cmap_out+cmap_out+[error_cmap],
            #                             cols_per_row=3,
            #                             # title=F"{output_fields[0]} RMSE: {np.mean(rmse_cnn):0.5f} m.")
            #                             title=F"SSH RMSE: {np.mean(rmse_cnn):0.5f} m.")

            print("DONE ALL FILES!!!!!!!!!!!!!")
    dic_summary = {
        "File": increment_files[start_test_idx:],
        "rmse": all_whole_rmse,
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
            maxcbar.append(.2)
            mincbar.append(-.2)
        elif c_field == "temp" or c_field == "sst" or c_field == "temp":
            maxcbar.append(np.nan)
            mincbar.append(np.nan)
        elif c_field == "salin" or c_field == "sss" or c_field == "sal":
            maxcbar.append(np.nan)
            mincbar.append(np.nan)
        elif c_field == "u-vel.":
            maxcbar.append(np.nan)
            mincbar.append(np.nan)
        elif c_field == "v-vel.":
            maxcbar.append(np.nan)
            mincbar.append(np.nan)
        elif c_field == "error":
            maxcbar.append(np.nan)
            mincbar.append(np.nan)
        else:
            maxcbar.append(np.nan)
            mincbar.append(np.nan)
    return mincbar, maxcbar


def denormalizeData(input, fields, data_type, norm_type):
    output = np.zeros(input.shape)
    for field_idx, c_field in enumerate(fields):
        # Denormalizing data...
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
    main()
