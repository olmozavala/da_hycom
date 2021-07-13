import os
from io_project.read_utils import generateXandY, normalizeData

from tensorflow.keras.utils import plot_model
from inout.io_netcdf import read_netcdf
from os.path import join
import numpy as np
import pandas as pd
import time
import cmocean

from img_viz.eoa_viz import EOAImageVisualizer
from config.MainConfig import get_prediction_params
from constants_proj.AI_proj_params import PredictionParams, ProjTrainingParams, PreprocParams
from models.modelSelector import select_2d_model
from models_proj.models import *
from constants.AI_params import TrainingParams, ModelParams
from img_viz.common import create_folder

from sklearn.metrics import mean_squared_error

from ExtraUtils.NamesManipulation import *
from eoas_utils.VizUtilsProj import chooseCMAP, getMinMaxPlot

# Rows and Columns to use
gcols = 1400
grows = 888



def main():

    config = get_prediction_params()
    # -------- For single model testing --------------
    # test_model(config)

    # -------- For all summary model testing --------------
    summary_file = "/data/HYCOM/DA_HYCOM_TSIS/SUMMARY/summary.csv"
    # summary_file = "/home/data/MURI/output/SUMMARY/summary.csv"
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
        model_fields = config[ProjTrainingParams.fields_names]
        obs_fields = config[ProjTrainingParams.fields_names_obs]
        print(F"Input fields: {model_fields}, {obs_fields}, {var_fields}")
        # config[ProjTrainingParams.fields_names] = model_fields
        # config[ProjTrainingParams.fields_names_obs] = obs_fields
        config[ProjTrainingParams.fields_names_var] = var_fields
        # Setting BBOX
        config[ModelParams.INPUT_SIZE][0] = grows
        config[ModelParams.INPUT_SIZE][1] = gcols
        config[ModelParams.INPUT_SIZE][2] = len(model_fields) + len(obs_fields) + len(var_fields)
        # Setting output vars
        output_fields = getOutputFields(name)
        config[ProjTrainingParams.output_fields] = output_fields
        # Setting model weights file
        config[PredictionParams.model_weights_file] = model["Path"]
        print(F"Model's weight file: {model['Path']}")
        # Set the name of the network
        run_name = name.replace(".hdf5", "")
        config[TrainingParams.config_name] = run_name
        test_model(config)

    # In this case we test all the best models

def test_model(config):
    input_folder = config[PredictionParams.input_folder]
    output_folder = config[PredictionParams.output_folder]
    output_fields = config[ProjTrainingParams.output_fields]
    model_weights_file = config[PredictionParams.model_weights_file]
    output_imgs_folder = config[PredictionParams.output_imgs_folder]
    field_names_model = config[ProjTrainingParams.fields_names]
    field_names_obs = config[ProjTrainingParams.fields_names_obs]
    rows = config[ProjTrainingParams.rows]
    cols = config[ProjTrainingParams.cols]
    run_name = config[TrainingParams.config_name]
    norm_type = config[ProjTrainingParams.norm_type]

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
    all_files = os.listdir(input_folder)
    all_files.sort()
    model_files = np.array([x for x in all_files if x.startswith('model')])

    z_layers = [0]
    var_file = join(input_folder, "cov_mat", "tops_ias_std.nc")
    field_names_std = config[ProjTrainingParams.fields_names_var]
    if len(field_names_std) > 0:
        input_fields_std = read_netcdf(var_file, field_names_std, z_layers)
    else:
        input_fields_std = []

    cmap_out = chooseCMAP(output_fields)
    cmap_model = chooseCMAP(field_names_model)
    cmap_obs = chooseCMAP(field_names_obs)
    cmap_std = chooseCMAP(field_names_std)

    tot_rows = 891
    tot_cols = 1401

    all_whole_mean_times = []
    all_whole_sum_times = []
    all_whole_rmse = []

    # np.random.shuffle(model_files)  # TODO this is only for testing
    for id_file, c_file in enumerate(model_files):
        # Find current and next date
        year = int(c_file.split('_')[1])
        day_of_year = int(c_file.split('_')[2].split('.')[0])

        model_file = join(input_folder, F'model_{year}_{day_of_year:03d}.nc')
        inc_file = join(input_folder,F'increment_{year}_{day_of_year:03d}.nc')
        obs_file = join(input_folder,F'obs_{year}_{day_of_year:03d}.nc')

        # *********************** Reading files **************************
        input_fields_model = read_netcdf(model_file, field_names_model, z_layers)
        input_fields_obs = read_netcdf(obs_file, field_names_obs, z_layers)
        output_field_increment = read_netcdf(inc_file, output_fields, z_layers)

        # ******************* Normalizing and Cropping Data *******************
        this_file_times = []

        try:
            perc_ocean = .01
            input_data, y_data = generateXandY(input_fields_model, input_fields_obs, input_fields_std, output_field_increment,
                                               field_names_model, field_names_obs, field_names_std, output_fields,
                                               0, 0, grows, gcols, norm_type=norm_type, perc_ocean=perc_ocean)
        except Exception as e:
            print(F"Exception {e}")

        # ******************* Replacing nan values *********
        # We set a value of 0.5 on the land. Trying a new loss function that do not takes into account land
        input_data_nans = np.isnan(input_data)
        input_data = np.nan_to_num(input_data, nan=0)
        y_data = np.nan_to_num(y_data, nan=-0.5)

        X = np.expand_dims(input_data, axis=0)
        Y = np.expand_dims(y_data, axis=0)

        # Make the prediction of the network
        start = time.time()
        output_nn_original = model.predict(X, verbose=1)
        toc = time.time() - start
        this_file_times.append(toc)

        # Make nan all values inside the land
        land_indexes = Y == -0.5
        output_nn_original[land_indexes] = np.nan

        # ==== Denormalizingallinput and outputs
        denorm_cnn_output = denormalizeData(output_nn_original, output_fields, PreprocParams.type_inc, norm_type)
        denorm_y = denormalizeData(Y, output_fields, PreprocParams.type_inc, norm_type)
        input_types = [PreprocParams.type_model for i in input_fields_model] + [PreprocParams.type_obs for i in input_fields_obs] + [PreprocParams.type_std for i in input_fields_std]
        denorm_input = denormalizeData(input_data, field_names_model+field_names_obs+field_names_std, input_types, norm_type)

        # Recover the original land areas, they are lost after denormalization
        denorm_y[land_indexes] = np.nan

        # Remove the 'extra dimension'
        denorm_cnn_output = np.squeeze(denorm_cnn_output)
        denorm_y = np.squeeze(denorm_y)
        whole_cnn = denorm_cnn_output# Add the the 'whole prediction'
        whole_y = denorm_y # Add the the 'whole prediction'

        if len(denorm_cnn_output.shape) == 2: # In this case we only had one output and we need to make it 'array' to plot
            denorm_cnn_output = np.expand_dims(denorm_cnn_output, axis=2)
            denorm_y = np.expand_dims(denorm_y, axis=2)

        # Compute RMSE
        # rmse_cnn = np.zeros(len(output_fields))
        # for i in range(len(output_fields)):
        #     ocean_indexes = np.logical_not(np.isnan(denorm_y[:,:,i]))
        #     rmse_cnn[i] = np.sqrt(mean_squared_error(denorm_cnn_output[:,:,i][ocean_indexes], denorm_y[:,:,i][ocean_indexes]))


        # ================== DISPLAYS ALL INPUTS AND OUTPUTS DENORMALIZED ===================
        # Adding back mask to all the input variables
        denorm_input[input_data_nans] = np.nan

        # ======= Plots whole output with RMSE
        mincbar = np.nanmin(whole_y)
        maxcbar = np.nanmax(whole_y)
        error = whole_y - whole_cnn
        mincbarerror = np.nanmin(error)
        maxcbarerror = np.nanmax(error)
        no_zero_ids = np.count_nonzero(whole_cnn)

        if output_fields[0] == 'srfhgt': # This should only be for SSH to adjust the units
            whole_cnn /= 9.81
            whole_y = np.array(whole_y)/9.81

        rmse_cnn = np.sqrt( np.nansum( (whole_y - whole_cnn)**2 )/no_zero_ids)

        all_whole_rmse.append(rmse_cnn)
        all_whole_mean_times.append(np.mean(np.array(this_file_times)))
        all_whole_sum_times.append(np.sum(np.array(this_file_times)))

        # if day_of_year == 353: # Plot 10% of the times
        if True: # Plot 10% of the times

            # viz_obj = EOAImageVisualizer(output_folder=output_imgs_folder, disp_images=False, mincbar=mincbar, maxcbar=maxcbar)
            viz_obj = EOAImageVisualizer(output_folder=output_imgs_folder, disp_images=False)


            # viz_obj.plot_2d_data_np_raw(np.concatenate((input_data.swapaxes(0,2), Y[0,:,:,:].swapaxes(0,2), output_nn_original[0,:,:,:].swapaxes(0,2))),
            viz_obj.plot_2d_data_np_raw(np.concatenate((denorm_input.swapaxes(0,2), denorm_y.swapaxes(0,2), denorm_cnn_output.swapaxes(0,2))),
                                        var_names=[F"in_model_{x}" for x in field_names_model] +
                                                  [F"in_obs_{x}" for x in field_names_obs] +
                                                  [F"in_var_{x}" for x in field_names_std] +
                                                  [F"out_inc_{x}" for x in output_fields] +
                                                  [F"cnn_{x}" for x in output_fields],
                                        file_name=F"Global_Input_and_CNN_{c_file}",
                                        rot_90=True,
                                        cmap=cmap_model+cmap_obs+cmap_std+cmap_out+cmap_out,
                                        cols_per_row=len(field_names_model),
                                        title=F"Input data: {field_names_model} and obs {field_names_obs}, increment {output_fields}, cnn {output_fields}")

            minmax = getMinMaxPlot(output_fields)[0]
            viz_obj = EOAImageVisualizer(output_folder=output_imgs_folder, disp_images=False,
                                         # mincbar=mincbar + mincbar + mincbarerror,
                                         # maxcbar=maxcbar + maxcbar + maxcbarerror)
                                         # mincbar=[minmax[0], minmax[0], max(minmax[0],-1)],
                                         # maxcbar=[minmax[1], minmax[1], min(minmax[1],1)])
                                        mincbar=[minmax[0], minmax[0], -1],
                                        maxcbar=[minmax[1], minmax[1], 1])

            # ================== Displays CNN and TSIS with RMSE ================
            error_cmap = cmocean.cm.diff
            viz_obj.output_folder = join(output_imgs_folder,'WholeOutput_CNN_TSIS')
            viz_obj.plot_2d_data_np_raw([np.flip(whole_cnn, axis=0), np.flip(whole_y, axis=0), np.flip(error, axis=0)],
                                        # var_names=[F"CNN INC {x}" for x in output_fields] + [F"TSIS INC {x}" for x in output_fields] + [F'TSIS - CNN (Mean RMSE {rmse_cnn:0.4f} m)'],
                                        var_names=[F"CNN increment SSH" for x in output_fields] + [F"TSIS increment SSH" for x in output_fields] + [F'TSIS - CNN \n (Mean RMSE {rmse_cnn:0.4f} m)'],
                                        file_name=F"Global_WholeOutput_CNN_TSIS_{c_file}",
                                        rot_90=False,
                                        cmap=cmap_out+cmap_out+[error_cmap],
                                        cols_per_row=3,
                                        # title=F"{output_fields[0]} RMSE: {np.mean(rmse_cnn):0.5f} m.")
                                        title=F"SSH RMSE: {np.mean(rmse_cnn):0.5f} m.")

            print("DONE ALL FILES!!!!!!!!!!!!!")
    dic_summary = {
        "File": model_files,
        "rmse": all_whole_rmse,
        "times mean": all_whole_mean_times,
        "times sum": all_whole_sum_times,
    }
    df = pd.DataFrame.from_dict(dic_summary)
    df.to_csv(join(output_imgs_folder, "Global_RMSE_and_times.csv"))


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
