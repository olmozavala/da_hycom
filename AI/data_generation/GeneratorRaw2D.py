# External
import numpy as np
from os.path import join, exists
from datetime import datetime,timedelta
import os, sys
import pickle
import tensorflow as tf

# Disable GPU usage
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TensorFlow logging

# Force TensorFlow to use CPU
# tf.config.set_visible_devices([], 'GPU')

# Disable eager execution
# tf.compat.v1.disable_eager_execution()

# Configure TensorFlow to not use GPU memory
# physical_devices = tf.config.list_physical_devices('GPU')
# try:
    # tf.config.experimental.set_memory_growth(physical_devices[0], True)
# except:
    # pass

# Common
sys.path.append("/unity/f1/ozavala/CODE/da_hycom/")
sys.path.append("/unity/f1/ozavala/CODE/da_hycom/eoas_pyutils/")
sys.path.append("/unity/f1/ozavala/CODE/da_hycom/eoas_pyutils/hycom_utils/python")
# For testing this file
from config.MainConfig_2D import get_training
from config.PreprocConfig import get_preproc_config

from viz_utils.eoa_viz import EOAImageVisualizer
from io_utils.io_netcdf import read_netcdf, read_netcdf_xr
from hycom.io import read_hycom_fields
from ai_common.constants.AI_params import TrainingParams,AiModels, ModelParams
# From project
from io_project.read_utils import generateXandY2D, generateXandYMulti, get_date_from_preproc_filename
from constants_proj.AI_proj_params import ProjTrainingParams, PreprocParams

def data_gen_from_raw(config, preproc_config, ids, field_names, obs_field_names, output_fields, z_layers=[0],
                      examples_per_figure=10, perc_ocean=0, composite_field_names=[], batch_size=1, uselatlon=False):
    """
    This generator should generate X and Y for a CNN
    :param path:
    :param file_names:
    :return:
    """
    ex_id = -1
    np.random.shuffle(ids)

    if config[ProjTrainingParams.YEARS] == "2009_2010":
        test_years = False
        print("Generating data for 2009 and 2010")
    elif config[ProjTrainingParams.YEARS] == "2002_2006":
        test_years = True
        print("Generating data for 2002 and 2006")

    rows = config[ProjTrainingParams.rows]
    cols = config[ProjTrainingParams.cols]

    output_folder = '/unity/f1/ozavala/OUTPUTS/DA_HYCOM_TSIS/ALL_INPUT/'
    # Check the file doesn't exist

    if test_years:
        gen_file_name = f"2002_2006_{rows}_{cols}_IN_{'_'.join(field_names)}_OBS_{'_'.join(obs_field_names)}_OUT_{'_'.join(output_fields)}_PERC_{str(int(perc_ocean*10))}.pkl"
    else:
        gen_file_name = f"2009_2010_{rows}_{cols}_IN_{'_'.join(field_names)}_OBS_{'_'.join(obs_field_names)}_OUT_{'_'.join(output_fields)}_PERC_{str(int(perc_ocean*10))}.pkl"

    print(f"Inside the generator, searching for file: {gen_file_name}")

    if exists(join(output_folder, f"X_{gen_file_name}")) and exists(join(output_folder, f'Y_{gen_file_name}')):
        print(f"The input files already exist. Loading them.... (X_{gen_file_name})")
        with open(join(output_folder, f'X_{gen_file_name}'), 'rb') as f:
            X_all = pickle.load(f)
        with open(join(output_folder, f'Y_{gen_file_name}'), 'rb') as f:
            Y_all = pickle.load(f)
        print("Done!")
    else:
        input_folder_background =   preproc_config[PreprocParams.input_folder_hycom]
        input_folder_increment =    preproc_config[PreprocParams.input_folder_tsis]
        input_folder_observations = preproc_config[PreprocParams.input_folder_obs]

        perc_ocean = config[ProjTrainingParams.perc_ocean]

        # --------- Reading all file names --------------
        if test_years:
            increment_files = np.array([join(input_folder_increment, x).replace(".a", "") for x in os.listdir(input_folder_increment) 
                                        if x.endswith('.a') and x.find('001_18') == -1 and (x.find('2002') != -1 or x.find('2006') != -1)])
        else:
            # --------- Reading only 2009 and 2010 (default training)
            increment_files = np.array([join(input_folder_increment, x).replace(".a", "") for x in os.listdir(input_folder_increment) 
                                        if x.endswith('.a') and x.find('001_18') == -1 and (x.find('2009') != -1 or x.find('2010') != -1)])
        increment_files.sort()
        # increment_files = increment_files[ids]  # Just reading the desired ids

        # To avoid problem with matching files we will obtain the date from the increment file and manually searh for the
        # corresponding background and observation files
        background_files = []
        obs_files = []

        for f_idx, c_file in enumerate(increment_files):
            sp_name = c_file.split("/")[-1].split(".")[1]
            c_datetime = datetime.strptime(sp_name, "%Y_%j_18")
            c_datetime_next_day = c_datetime + timedelta(days=1)

            background_files.append(join(input_folder_background, F"022_archv.{c_datetime.strftime('%Y_%j')}_18.a")) # For 2002, 2006, 2009 and 2010
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
        z_layers = [0] # Because it is 2D we only focus on zlayer =0
        norm_type = config[ProjTrainingParams.norm_type]
        #
        # # Read the variance of selected
        # var_field_names = config[ProjTrainingParams.fields_names_var]
        # if len(var_field_names) > 0:
        #     input_fields_var = read_netcdf(var_file, var_field_names, z_layers)
        # else:
        #     input_fields_var = []

        print(f"Preloading all data. Total number of examples: {len(increment_files)}. Total RAM in skynet 1TB")
        X_all = []
        Y_all = []
        for idx in range(len(increment_files)):
            print("Preloading file ", idx)
            increment_file_name = increment_files[idx]
            obs_file_name = obs_files[idx]
            model_file_name = background_files[idx]

            # *********************** Reading files **************************
            input_fields_model = read_hycom_fields(model_file_name, field_names, z_layers)
            input_fields_obs = read_netcdf_xr(obs_file_name, obs_field_names, z_layers)
            output_field_increment = read_hycom_fields(increment_file_name, output_fields, z_layers)

            if rows < 300: # In this case we assume we are looking at smaller BBOX
                succ_attempts = 0
                print(f"Generating multiple examples from the same file {succ_attempts}/{examples_per_figure}")
                # ------------------------------------- Generating multiple examples from the same file ---------------------
                while succ_attempts < examples_per_figure: # Generate multiple images from the same file
                    start_row = np.random.randint(0, 384 - rows)  # These hardcoded numbers come from the specific size of these files
                    start_col = np.random.randint(0, 525 - cols)

                    try:
                        input_data, y_data = generateXandY2D(input_fields_model, input_fields_obs, [], output_field_increment,
                                                        field_names+composite_field_names, obs_field_names, [], output_fields,
                                                        start_row, start_col, rows, cols, norm_type=norm_type, perc_ocean=perc_ocean)

                        # Making all land pixels to 0
                        input_data = np.nan_to_num(input_data, nan=0)
                        y_data = np.nan_to_num(y_data, nan=0)

                        X = np.expand_dims(input_data, axis=0)
                        Y = np.expand_dims(y_data, axis=0)

                        X[0,:,:,2] = np.nan_to_num(X[0,:,:,2], 0 )
                        X[0,:,:,3] = np.nan_to_num(X[0,:,:,3], 0 )

                        X_all.append(X)
                        Y_all.append(Y)

                        succ_attempts += 1
                    except Exception as e:
                        print(F"Failed for {model_file_name} {start_row}-{start_col}: {e}")
                        continue
                print(f"Succesfully generated {succ_attempts} examples from the same file")

            else:  # In this case we assume we really want to use the whole domain
                start_row = 384 - rows  # These hardcoded numbers come from the specific size of these files
                start_col = 525 - cols

            try:
                input_data, y_data = generateXandY2D(input_fields_model, input_fields_obs, [], output_field_increment,
                                                    field_names+composite_field_names, obs_field_names, [], output_fields,
                                                    start_row, start_col, rows, cols, norm_type=norm_type, perc_ocean=perc_ocean, uselatlon=uselatlon)
            except Exception as e:
                print(F"Failed for {model_file_name} {start_row}-{start_col}: {e}")
                continue

            # Making all land pixels to 0
            input_data = np.nan_to_num(input_data, nan=0)
            y_data = np.nan_to_num(y_data, nan=0)

            X = np.expand_dims(input_data, axis=0)
            Y = np.expand_dims(y_data, axis=0)

            X[0,:,:,2] = np.nan_to_num(X[0,:,:,2], 0 )
            X[0,:,:,3] = np.nan_to_num(X[0,:,:,3], 0 )
            X_all.append(X)
            Y_all.append(Y)

            single_size = X.nbytes / (1024**3)
            size_gb_all = single_size * len(X_all)
            print(f"Size of the array in memory: {2*size_gb_all:.6f} GB")


        # Save X and Y as two huge pickle files
        x_file = join(output_folder, f'X_{gen_file_name}')
        y_file = join(output_folder, f'Y_{gen_file_name}')
        with open(x_file, 'wb') as f:
            pickle.dump(X_all, f)
        with open(y_file, 'wb') as f:
            pickle.dump(Y_all, f)
    
    num_samples = len(ids)
    sub_ids = np.arange(0,len(ids)) # Reset and shuffle the ids
    np.random.shuffle(sub_ids) # We shuffle the folders every time we have tested all the examples
    print(f"Shape of the input data: {X_all[0].shape}")
    print(f"Shape of the output data: {Y_all[0].shape}")

    while True:
        # These lines are for sequential selection
        if ex_id < (len(ids) - 1): # We are not supporting batch processing right now
            ex_id += 1
        else:
            ex_id = 0
            np.random.shuffle(sub_ids) # We shuffle the folders every time we have tested all the examples

        for start in range(0, num_samples, batch_size):
            try:
                end = min(start + batch_size, num_samples)
                batch_indices = ids[start:end]
                
                # Convert batch_indices to list so that they can be used to index lists
                X_batch = np.array([X_all[i] for i in batch_indices])
                X_batch = np.concatenate(X_batch)  # Combine them into

                Y_batch = np.array([Y_all[i] for i in batch_indices])
                Y_batch = np.concatenate(Y_batch)  # Combine them into

                # --------------- Just for debugging Plotting input and output---------------------------
                # if False:
                    # import matplotlib.pyplot as plt
                    # from scipy.ndimage import convolve, gaussian_filter
                    # import cmocean
                    # plot_idx = 0
                    # cur_x = X_all[plot_idx]
                    # cur_y = Y_all[plot_idx]
                    # mincbar = np.nanmin(cur_x)
                    # maxcbar = np.nanmax(cur_y)
                    # output_folder = "/unity/f1/ozavala/OUTPUTS/DA_HYCOM_TSIS/TrainingSkynetPaperReviews/PaperPlots/"
                    # # Replace nan to 0
                    # cur_x[0,:,:,2] = np.nan_to_num(cur_x[0,:,:,2], 0 )
                    # X[0,:,:,3] = np.nan_to_num(X[0,:,:,3], 0 )
                    # # Apply gaussian filter
                    # X[0,:,:,2] = gaussian_filter(X[0,:,:,2], 2)
                    # X[0,:,:,3] = gaussian_filter(X[0,:,:,3], 2)

                    # viz_obj = EOAImageVisualizer(output_folder=output_folder, disp_images=False)
                    # viz_obj.plot_2d_data_np(np.rollaxis(X[0,:,:,:],2,0),
                    #                             var_names=[F"in_model_{x}" for x in field_names+composite_field_names] +
                    #                                         [F"in_obs_{x}" for x in obs_field_names],
                    #                             cmap=[cmocean.cm.thermal, cmocean.cm.curl,cmocean.cm.curl,cmocean.cm.thermal,cmocean.cm.curl],
                    #                             # cmap=cmocean.cm.curl,
                    #                             file_name_prefix=F"{plot_idx}_{start_col}_{start_row}",
                    #                             title=F"")

                    # viz_obj.plot_2d_data_np(np.rollaxis(Y[0,:,:,:],2,0),
                                            # var_names=[F"out_model_{x}" for x in output_fields] ,
                                            # file_name_prefix=F"{plot_idx}_{start_col}_{start_row}_out",
                                            # title=F"{increment_file_name}")

                yield X_batch, Y_batch


            except Exception as e:
                print(F"----- Not able to generate for file number ()from batch):   ERROR: {e}")

def get_dataset(config, preproc_config, ids, field_names, obs_field_names, output_fields, z_layers=[0],
                      examples_per_figure=10, perc_ocean=0, composite_field_names=[], batch_size=1):

    # Convert config and preproc_config to dictionaries
    # dataset = tf.data.Dataset.from_generator(
    #     lambda: data_gen_from_raw(config, preproc_config, ids, field_names, obs_field_names, output_fields, z_layers,
    #                               examples_per_figure, perc_ocean, composite_field_names, batch_size),
    #     output_signature=(
    #         tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32),
    #         tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32)
    #     )
    # )
    # return dataset
    dataset = tf.data.Dataset.from_generator(data_gen_from_raw, args=[config, preproc_config, ids, field_names, obs_field_names, output_fields, z_layers,
                      examples_per_figure, perc_ocean, composite_field_names, batch_size],
                                            output_signature=(
                                                tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32),
                                                tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32)
                                            ))
    return dataset

# Test the dataset
if __name__ == "__main__":
    # from config.MainConfig_2D import get_training, get_prediction_params
    # from config.PreprocConfig import get_preproc_config

    config = get_training()
    preproc_config = get_preproc_config()
    # config = {}
    # preproc_config = {}

    # config[ProjTrainingParams.rows] = 384
    # config[ProjTrainingParams.cols] = 520
    # config[ProjTrainingParams.perc_ocean] = 0
    # config[ProjTrainingParams.network_type] = NetworkTypes.UNET

    # preproc_config[PreprocParams.input_folder_hycom] = "/unity/g1/abozec/TSIS/GOMb0.04/expt_02.2/data"
    # preproc_config[PreprocParams.input_folder_tsis] = "/unity/g1/abozec/TSIS/GOMb0.04/expt_02.2/incup"
    # preproc_config[PreprocParams.input_folder_obs] = "/unity/g1/abozec/TSIS/GOMb0.04/obs/qcobs_roif"

    ids = np.arange(0, 10)
    # field_names = ["u-vel.", "v-vel.", "temp", "salin", "thknss", "srfhgt", "mix_dpth"]
    field_names = ["srfhgt", "temp"]
    obs_field_names = ["ssh", "ssh_err", "sst", "sst_err"]
    # output_fields = ["srfhgt", "temp", "u-vel.", "v-vel.", "salin", "thknss", "srfhgt"]
    output_fields = ["srfhgt", "temp"]
    z_layers = [0]
    examples_per_figure = 1
    perc_ocean = 0
    composite_field_names = ["diff_ssh","diff_sst","topo"]
    batch_size = 1

    dataset = get_dataset(config, preproc_config, ids, field_names, obs_field_names, output_fields, z_layers,
                          examples_per_figure, perc_ocean, composite_field_names, batch_size)

    for X, Y in dataset:
        print(f"X shape: {X.shape}, Y shape: {Y.shape}")
        break