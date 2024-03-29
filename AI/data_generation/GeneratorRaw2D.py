# External
import numpy as np
from os.path import join
from datetime import datetime,timedelta
import os, sys
# Common
sys.path.append("eoas_pyutils/")
sys.path.append("hycom_utils/python")
from viz_utils.eoa_viz import EOAImageVisualizer
from io_utils.io_netcdf import read_netcdf, read_netcdf_xr
from hycom.io import read_hycom_fields
from ai_common.constants.AI_params import TrainingParams,AiModels, ModelParams
# From project
from io_project.read_utils import generateXandY2D, generateXandYMulti, get_date_from_preproc_filename
from constants_proj.AI_proj_params import ProjTrainingParams, PreprocParams

def data_gen_from_raw(config, preproc_config, ids, field_names, obs_field_names, output_fields, z_layers=[0],
                      examples_per_figure=10, perc_ocean=0, composite_field_names=[]):
    """
    This generator should generate X and Y for a CNN
    :param path:
    :param file_names:
    :return:
    """
    ex_id = -1
    np.random.shuffle(ids)

    input_folder_background =   preproc_config[PreprocParams.input_folder_hycom]
    input_folder_increment =    preproc_config[PreprocParams.input_folder_tsis]
    input_folder_observations = preproc_config[PreprocParams.input_folder_obs]

    perc_ocean = config[ProjTrainingParams.perc_ocean]

    # --------- Reading all file names --------------
    increment_files = np.array([join(input_folder_increment, x).replace(".a", "") for x in os.listdir(input_folder_increment) if x.endswith('.a')])
    increment_files.sort()
    increment_files = increment_files[ids]  # Just reading the desired ids

    sub_ids = np.arange(0,len(ids)) # Reset and shuffle the ids
    np.random.shuffle(sub_ids) # We shuffle the folders every time we have tested all the examples
    # To avoid problem with matching files we will obtain the date from the increment file and manually searh for the
    # corresponding background and observation files
    background_files = []
    obs_files = []

    for f_idx, c_file in enumerate(increment_files):
        sp_name = c_file.split("/")[-1].split(".")[1]
        c_datetime = datetime.strptime(sp_name, "%Y_%j_18")
        c_datetime_next_day = c_datetime + timedelta(days=1)

        background_files.append(join(input_folder_background, F"022_archv.{c_datetime.strftime('%Y_%j')}_18.a")) # For 2002 and 2006
        # background_files.append(join(input_folder_background, F"020_archv.{c_datetime.strftime('%Y_%j')}_18.a")) # For 2009 and 2010
        obs_files.append(join(input_folder_observations, F"tsis_obs_gomb4_{c_datetime_next_day.strftime('%Y%m%d')}00.nc"))
        # print(background_files[f_idx])
        assert os.path.exists(background_files[f_idx])
        # print(obs_files[f_idx])
        assert os.path.exists(obs_files[f_idx])

    background_files = np.array(background_files)
    obs_files = np.array(obs_files)

    # --------- Reading all file names --------------
    # background_files = np.array([join(input_folder_preproc, x) for x in all_files if x.startswith('model')])
    # var_file = join(input_folder_preproc, "cov_mat", "tops_ias_std.nc")
    #
    rows = config[ProjTrainingParams.rows]
    cols = config[ProjTrainingParams.cols]
    z_layers = [0] # Because it is 2D we only focus on zlayer =0
    norm_type = config[ProjTrainingParams.norm_type]
    #
    # # Read the variance of selected
    # var_field_names = config[ProjTrainingParams.fields_names_var]
    # if len(var_field_names) > 0:
    #     input_fields_var = read_netcdf(var_file, var_field_names, z_layers)
    # else:
    #     input_fields_var = []
    #
    while True:
        # These lines are for sequential selection
        if ex_id < (len(ids) - 1): # We are not supporting batch processing right now
            ex_id += 1
        else:
            ex_id = 0
            np.random.shuffle(sub_ids) # We shuffle the folders every time we have tested all the examples

        c_id = sub_ids[ex_id]
        try:
            increment_file_name = increment_files[c_id]
            obs_file_name = obs_files[c_id]
            model_file_name = background_files[c_id]

            # *********************** Reading files **************************
            input_fields_model = read_hycom_fields(model_file_name, field_names, z_layers)
            input_fields_obs = read_netcdf_xr(obs_file_name, obs_field_names, z_layers)
            output_field_increment = read_hycom_fields(increment_file_name, output_fields, z_layers)

            succ_attempts = 0
            while succ_attempts < examples_per_figure: # Generate multiple images from the same file
                if rows < 300: # In this case we assume we are looking at smaller BBOX
                    start_row = np.random.randint(0, 384 - rows)  # These hardcoded numbers come from the specific size of these files
                    start_col = np.random.randint(0, 525 - cols)
                else:  # In this case we assume we really want to use the whole domain
                    start_row = 384 - rows  # These hardcoded numbers come from the specific size of these files
                    start_col = 525 - cols

                try:
                    input_data, y_data = generateXandY2D(input_fields_model, input_fields_obs, [], output_field_increment,
                                                       field_names+composite_field_names, obs_field_names, [], output_fields,
                                                        start_row, start_col, rows, cols, norm_type=norm_type, perc_ocean=perc_ocean)
                    # start_row, start_col, rows, cols, norm_type=PreprocParams.no_norm, perc_ocean=perc_ocean)
                except Exception as e:
                    # print(F"Failed for {model_file_name} {start_row}-{start_col}: {e}")
                    continue

                succ_attempts += 1

                # Making all land pixels to 0
                input_data = np.nan_to_num(input_data, nan=0)
                y_data = np.nan_to_num(y_data, nan=0)

                X = np.expand_dims(input_data, axis=0)
                Y = np.expand_dims(y_data, axis=0)
                # --------------- Just for debugging Plotting input and output---------------------------
                import matplotlib.pyplot as plt
                from scipy.ndimage import convolve, gaussian_filter
                import cmocean
                # mincbar = np.nanmin(input_data)
                # maxcbar = np.nanmax(input_data)
                output_folder = config[TrainingParams.output_folder]
                # Replace nan to 0
                X[0,:,:,2] = np.nan_to_num(X[0,:,:,2], 0 )
                X[0,:,:,3] = np.nan_to_num(X[0,:,:,3], 0 )
                # Apply gaussian filter
                # X[0,:,:,2] = gaussian_filter(X[0,:,:,2], 2)
                # X[0,:,:,3] = gaussian_filter(X[0,:,:,3], 2)

                viz_obj = EOAImageVisualizer(output_folder=join(output_folder, "training_imgs"), disp_images=False)
                viz_obj.plot_2d_data_np(np.rollaxis(X[0,:,:,:],2,0),
                                            var_names=[F"in_model_{x}" for x in field_names+composite_field_names] +
                                                      [F"in_obs_{x}" for x in obs_field_names],
                                            cmap=[cmocean.cm.thermal, cmocean.cm.curl,cmocean.cm.curl,cmocean.cm.thermal,cmocean.cm.curl],
                                            # cmap=cmocean.cm.curl,
                                            file_name_prefix=F"{c_id}_{start_col}_{start_row}",
                                            title=F"")

                viz_obj.plot_2d_data_np(np.rollaxis(Y[0,:,:,:],2,0),
                                        var_names=[F"out_model_{x}" for x in output_fields] ,
                                        file_name_prefix=F"{c_id}_{start_col}_{start_row}_out",
                                        title=F"{increment_file_name}")

                yield X, Y
                # yield [np.zeros((1, 384, 520, 24, 8))], [np.zeros((1, 384, 520, 24, 6))]
        except Exception as e:
            print(F"----- Not able to generate for file number (from batch):   ERROR: {e}")

