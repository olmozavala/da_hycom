import numpy as np
from inout.io_netcdf import read_netcdf
from io_project.read_utils import generateXandY, generateXandYMulti, get_date_from_preproc_filename
from os.path import join, exists
from constants_proj.AI_proj_params import MAX_MODEL, MAX_OBS, MIN_OBS, MIN_MODEL, MAX_INCREMENT, MIN_INCREMENT
import os
from constants_proj.AI_proj_params import ProjTrainingParams
from constants.AI_params import TrainingParams
from img_viz.eoa_viz import EOAImageVisualizer
from constants.AI_params import AiModels, ModelParams

def data_gen_from_preproc(input_folder_preproc,  config, ids, field_names, obs_field_names, output_fields, z_layers=[0]):
    """
    This generator should generate X and Y for a CNN
    :param path:
    :param file_names:
    :return:
    """
    ex_id = -1
    np.random.shuffle(ids)
    batch_size = config[TrainingParams.batch_size]

    all_files = os.listdir(input_folder_preproc)
    obs_files = np.array([join(input_folder_preproc, x) for x in all_files if x.startswith('obs')])
    increment_files = np.array([join(input_folder_preproc, x) for x in all_files if x.startswith('increment')])
    model_files = np.array([join(input_folder_preproc, x) for x in all_files if x.startswith('model')])
    var_file = join(input_folder_preproc, "cov_mat", "tops_ias_std.nc")
    obs_files.sort()
    increment_files.sort()
    model_files.sort()

    rows = config[ProjTrainingParams.rows]
    cols = config[ProjTrainingParams.cols]
    norm_type = config[ProjTrainingParams.norm_type]

    # Read the variance of selected
    var_field_names = config[ProjTrainingParams.fields_names_var]
    if len(var_field_names) > 0:
        input_fields_var = read_netcdf(var_file, var_field_names, z_layers)
    else:
        input_fields_var = []

    while True:
        # These lines are for sequential selection
        if ex_id < (len(ids) - 1): # We are not supporting batch processing right now
            ex_id += 1
        else:
            ex_id = 0
            np.random.shuffle(ids) # We shuffle the folders every time we have tested all the examples

        c_id = ids[ex_id]
        try:
            output_file_name = increment_files[c_id]
            obs_file_name = obs_files[c_id]
            model_file_name = model_files[c_id]

            # Needs to validate that all the files are from the same date
            model_file_year, model_file_day = get_date_from_preproc_filename(model_file_name)
            obs_file_year, obs_file_day = get_date_from_preproc_filename(obs_file_name)
            output_file_year, output_file_day = get_date_from_preproc_filename(output_file_name)

            if (model_file_day != obs_file_day) or (model_file_day != output_file_day) or\
                    (model_file_year != obs_file_year) or (model_file_year != output_file_year):
               print(F"The year and day do not correspond between the files: {output_file_name}, {model_file_name}, {obs_file_name}")
               exit()

            # If any file doesn't exist, jump to the next example
            if not(exists(output_file_name)):
                print(F"File doesn't exist: {output_file_name}")
                continue

            # *********************** Reading files **************************
            input_fields_model = read_netcdf(model_file_name, field_names, z_layers)
            input_fields_obs = read_netcdf(obs_file_name, obs_field_names, z_layers)
            output_field_increment = read_netcdf(output_file_name, output_fields, z_layers)

            succ_attempts = 0
            while succ_attempts < batch_size:
                start_row = np.random.randint(0, 891 - rows)  # These hardcoded numbers come from the specific size of these files
                start_col = np.random.randint(0, 1401 - cols)

                try:
                    perc_ocean = 0.99
                    if config[ModelParams.MODEL] == AiModels.UNET_2D_MULTISTREAMS:
                        input_data, y_data = generateXandYMulti(input_fields_model, input_fields_obs, input_fields_var, output_field_increment,
                                                                field_names, obs_field_names, var_field_names, output_fields,
                                                           start_row, start_col, rows, cols, norm_type=norm_type)
                    else:
                        input_data, y_data = generateXandY(input_fields_model, input_fields_obs, input_fields_var, output_field_increment,
                                                           field_names, obs_field_names, var_field_names, output_fields,
                                                           start_row, start_col, rows, cols, norm_type=norm_type, perc_ocean=perc_ocean)

                except Exception as e:
                    # print(F"Failed for {model_file_name} row:{start_row} col:{start_col}: {e}")
                    continue

                succ_attempts += 1

                # We set a value of 0.5 on the land. Trying a new loss function that do not takes into account land
                input_data = np.nan_to_num(input_data, nan=0)
                y_data = np.nan_to_num(y_data, nan=-0.5)
                # input_data = np.nan_to_num(input_data, nan=-1000)
                # y_data = np.nan_to_num(y_data, nan=-1000)
                # input_data = np.nan_to_num(input_data, nan=0)
                # y_data = np.nan_to_num(y_data, nan=0)

                if config[ModelParams.MODEL] == AiModels.UNET_2D_MULTISTREAMS:
                    X = [np.expand_dims(x, axis=0) for x in input_data]
                else:
                    X = np.expand_dims(input_data, axis=0)
                Y = np.expand_dims(y_data, axis=0)

                # --------------- Just for debugging Plotting input and output---------------------------
                # import matplotlib.pyplot as plt
                # import pylab
                # # mincbar = np.nanmin(input_data)
                # # maxcbar = np.nanmax(input_data)
                #
                # # viz_obj = EOAImageVisualizer(output_folder=join(input_folder_preproc, "training_imgs"), disp_images=False, mincbar=mincbar, maxcbar=maxcbar)
                # viz_obj = EOAImageVisualizer(output_folder=join(input_folder_preproc, "training_imgs"), disp_images=False)
                # #
                # # viz_obj.plot_2d_data_np_raw(np.concatenate((input_data.swapaxes(0,2), y_data.swapaxes(0,2))),
                # viz_obj.plot_2d_data_np_raw(np.concatenate((X[0,:,:,:].swapaxes(0,2), Y[0,:,:,:].swapaxes(0,2))),
                #                             var_names=[F"in_model_{x}" for x in field_names] +
                #                                       [F"in_obs_{x}" for x in obs_field_names]+
                #                                       [F"out_inc_{x}" for x in output_fields],
                #                             rot_90=True,
                #                             file_name=F"{model_file_year}_{model_file_day}_{start_col}_{start_row}",
                #                             title=F"Input data: {field_names} and {obs_field_names}, output {output_fields}")

                yield X, Y
                # yield [np.zeros((1,160,160,1)) for x in range(7)], Y
        except Exception as e:
            print(F"----- Not able to generate for file number (from batch):  {succ_attempts} ERROR: ", str(e))

