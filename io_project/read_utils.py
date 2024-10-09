
from hycom.io import read_hycom_coords
from io_utils.dates_utils import get_days_from_month
from constants_proj.AI_proj_params import PreprocParams
import pandas as pd
from pandas import DataFrame
import numpy.ma as ma
from constants_proj.AI_proj_params import MAX_MODEL, MAX_OBS, MIN_OBS, MIN_MODEL, MAX_INCREMENT, MIN_INCREMENT, MIN_STDVAR, MAX_STDVAR
from os.path import join, isfile
import os
import numpy as np
import xarray as xr

def get_preproc_increment_files(input_folder):
    all_files = os.listdir(input_folder)
    all_files.sort()
    valid_files = [file for file in all_files if file.find('increment_') != -1]
    valid_paths = [join(input_folder, file) for file in valid_files]
    return np.array(valid_files), np.array(valid_paths)

def get_da_tot_days(input_folder, years, pies):
    """
    It should get all the days (files) for the specified year and path.
    :param input_folder:
    :param years:
    :return:
    """
    filtered_files = []
    corresponding_paths = []
    pies_txt = 'PIES' if pies else 'noPIES'
    for year in years:
        folder = join(input_folder, F'{str(year)}_{pies_txt}')
        all_files = os.listdir(folder)
        selected_files = [file for file in all_files if file.endswith('.a')]
        corresponding_paths+= [folder for file in selected_files]
        filtered_files += selected_files

    filtered_files_np = np.array(filtered_files)
    corresponding_paths_np = np.array(corresponding_paths)
    # Sorting the results
    filtered_files_np = np.sort(filtered_files)
    sorted_idxs = np.argsort(filtered_files)
    corresponding_paths_np = corresponding_paths_np[sorted_idxs]
    return filtered_files_np, corresponding_paths_np

def get_hycom_file_name(input_folder, year, month, day_idx=1):
    """
    This function obtains the complete path of the files for the specified month and year, stored by Dmitry with DA
    :param input_folder:
    :param year:
    :param month:
    :param day_idx: Indicates the position of the day of the year in the split by '_' of the file name
    :return: str array [file names], str array [file paths]
    """
    _, days_of_year = get_days_from_month(month)
    # folder = join(input_folder, F'{str(year)}')
    folder = input_folder
    all_files = os.listdir(folder)
    all_files = [x for x in all_files  if isfile(join(input_folder, x))]
    selected_files = [file for file in all_files if int(file.split('_')[day_idx]) in days_of_year]
    selected_files.sort()
    return selected_files, [join(folder,c_file) for c_file in selected_files]

def get_forecast_file_name(input_folder, year, month):
    """
    This function obtains the complete path of the files for the specified month and year, stored by Alec
    :param input_folder:
    :param year:
    :param month:
    :param pies:
    :return:
    """
    folder = join(input_folder, F'{year}{month:02d}')
    all_files = os.listdir(folder)
    all_files.sort()
    return all_files, [join(folder,c_file) for c_file in all_files]

def get_obs_file_names(input_folder, year, month):
    """
    This function obtains the complete path of the files for the specified month and year, observations
    :param input_folder:
    :param year:
    :param month:
    :param pies:
    :return:
    """
    # folder = join(input_folder, F'{pies_txt}')
    folder = input_folder
    all_files = os.listdir(folder)
    # all_files = [file for file in all_files if file.find('tsis_obs_ias') != -1] # Avoid additiona files
    all_files = [file for file in all_files if file.find('tsis_obs_gomb4') != -1] # Avoid additiona files
    selected_files = [file for file in all_files if (file.split('_')[3][0:6]) == F'{year}{month:02d}']
    selected_files.sort()
    return selected_files, [join(folder, c_file) for c_file in selected_files]

def normalizeData(data, field_name, data_type = PreprocParams.type_model, norm_type= PreprocParams.zero_one, normalize=True):
    """
    :param data: The data as a np array
    :param data_type: The type of data we are working on (model, obs or increment)
    :param norm_type: Which normalization are we doing (zero to one or mean 0 variance 1)
    :param normalize: if True it will normalize, if false it will denormalize
    :return:
    """
    if norm_type == PreprocParams.no_norm:
        return data

    if norm_type == PreprocParams.zero_one:
        if normalize:
            if data_type == PreprocParams.type_model:
                output_data = (data - MIN_MODEL[field_name]) /(MAX_MODEL[field_name] - MIN_MODEL[field_name])
            if data_type == PreprocParams.type_obs:
                output_data = (data.filled(np.nan) - MIN_OBS[field_name])/(MAX_OBS[field_name] - MIN_OBS[field_name])
            if data_type == PreprocParams.type_inc:
                output_data = (data - MIN_INCREMENT[field_name]) /(MAX_INCREMENT[field_name] - MIN_INCREMENT[field_name])
            if data_type == PreprocParams.type_std:
                output_data = (data - MIN_STDVAR[field_name]) /(MAX_STDVAR[field_name] - MIN_STDVAR[field_name])
        else:
            if data_type == PreprocParams.type_model:
                output_data = (data*(MAX_MODEL[field_name] - MIN_MODEL[field_name])) + MIN_MODEL[field_name]
            if data_type == PreprocParams.type_obs:
                output_data = (data*(MAX_OBS[field_name] - MIN_OBS[field_name])) + MIN_OBS[field_name]
            if data_type == PreprocParams.type_inc:
                output_data = (data*(MAX_INCREMENT[field_name] - MIN_INCREMENT[field_name])) + MIN_INCREMENT[field_name]
            if data_type == PreprocParams.type_std:
                output_data = (data*(MAX_STDVAR[field_name] - MIN_STDVAR[field_name])) + MIN_STDVAR[field_name]

        return output_data

    if norm_type == PreprocParams.mean_var:
        # First reading the model ones
        if data_type == PreprocParams.type_model or data_type == PreprocParams.type_inc:
            df = pd.read_csv("stats_background.csv")
        if data_type == PreprocParams.type_obs:
            df = pd.read_csv("stats_obs.csv")

        try:
            mean_val = df[df['model'] == 'mean'][field_name].item()
            std_val = df[df['model'] == 'std'][field_name].item()
            if normalize:
                output_data = (data - mean_val)/std_val
            else:
                output_data = (data*std_val) + mean_val

            if type(output_data) is np.ndarray:
                return output_data
            else:
                # return output_data.filled(np.nan)
                return output_data.data
        except Exception as e:
            print(f"Error from the normalization code: {e}")
            return data

# if norm_type == PreprocParams.mean_var:
    #     df = pd.read_csv("MIN_MAX_MEAN_STD_FINAL.csv")
    #     if data_type == PreprocParams.type_model:
    #         c_data = df[(df['TYPE'] == 'MODEL') & (df['Field'] == field_name)]
    #     if data_type == PreprocParams.type_obs:
    #         c_data = df[(df['TYPE'] == 'OBS') & (df['Field'] == field_name)]
    #     if data_type == PreprocParams.type_inc:
    #         c_data = df[(df['TYPE'] == 'INC') & (df['Field'] == field_name)]
        # min_val = c_data['MIN']
        # max_val = c_data['MAX']
        # mean_val = c_data['MEAN'].values[0]
        # std_val = c_data['STD'].values[0]
        # if normalize:
        #     output_data = (data - mean_val)/std_val
        # else:
        #     output_data = (data*std_val) + mean_val
        #
        # if type(output_data) is np.ndarray:
        #     return output_data
        # else:
        #     return output_data.filled(np.nan)

def generateXandYMulti(input_fields_model, input_fields_obs, input_fields_var, output_field_increment,
                       field_names, obs_field_names, var_field_names, output_fields,
                      start_row, start_col, rows, cols, norm_type = PreprocParams.mean_var):
    """
    This function will generate X and Y boxes depening on the required field names and bboxes
    :param input_fields_model:
    :param input_fields_obs:
    :param output_field_increment:
    :param field_names:
    :param obs_field_names:
    :param output_fields:
    :param start_row:
    :param start_col:
    :param rows:
    :param cols:
    :return:
    """
    num_fields = len(field_names) + len(obs_field_names) + len(var_field_names)
    input_data = []
    tot_output_fields = len(output_fields)
    y_data = np.zeros((rows, cols, tot_output_fields))

    end_row = start_row + rows
    end_col = start_col + cols
    id_field = 0 # Id of the input fields

    # ******* Adding the model fields for input ********
    for c_field in field_names:
        temp_data = input_fields_model[c_field][start_row:end_row, start_col:end_col]
        # For debugging
        # import matplotlib.pyplot as plt
        # plt.imshow(temp_data.data)
        # plt.title(c_field)
        # plt.show()
        # TODO couldn't find a better way to check that there are not 'land' pixels. IF YOU CHANGE IT YOU NEED TO BE SURE IT IS WORKING!!!!!!
        if len(temp_data.mask.shape) != 1: # It means it has at least some ocean values
            if temp_data.mask.min() == False or np.dtype('bool') != temp_data.mask.min(): # Making sure there is at leas some ocean pixels
                # This is a harder restriction, we force that 90% the pixels are ocean
                # if temp_data.count() > (.99*temp_data.shape[0]*temp_data.shape[1]):
                if temp_data.count() >= (temp_data.shape[0]*temp_data.shape[1]):
                    input_data.append(np.expand_dims(normalizeData(temp_data, c_field, data_type=PreprocParams.type_model,
                                                               norm_type=norm_type, normalize=True), axis=2))

                    if id_field == 0:
                        c_mask = ma.getmaskarray(temp_data)
                    id_field += 1
                else:
                    raise Exception("Not mostly ocean")
            else:
                raise Exception("Only land")
        else:
            raise Exception("Only land")

    # ******* Adding the observations fields for input ********
    for c_field in obs_field_names:
        temp_data = input_fields_obs[c_field][start_row:end_row, start_col:end_col]
        input_data.append(np.expand_dims(normalizeData(temp_data, c_field, data_type=PreprocParams.type_obs,
                                                   norm_type=norm_type, normalize=True), axis=2))
        id_field += 1

    # ******* Adding the variance fields for input ********
    for c_field in var_field_names:
        temp_data = input_fields_var[c_field][start_row:end_row, start_col:end_col]
        input_data.append(np.expand_dims(normalizeData(temp_data, c_field, data_type=PreprocParams.type_std,
                                                       norm_type=norm_type, normalize=True), axis=2))
        id_field += 1

    # ******************* Filling the 'output' data ***************
    id_field = 0
    for c_field in output_fields:
        temp_data = output_field_increment[c_field][start_row:end_row, start_col:end_col]
        # TODO this is a hack, for some reason the output field is not properly masked. So we force it to nan
        temp_data[c_mask] = np.nan
        y_data[:, :, id_field] = normalizeData(temp_data, c_field, data_type=PreprocParams.type_inc,
                                               norm_type=norm_type, normalize=True)
        # viz_obj = EOAImageVisualizer(output_folder=join(input_folder_preproc, "training_imgs"), disp_images=True)
        # viz_obj.plot_2d_data_np_raw(y_data.swapaxes(0,2),
        #                             var_names= [F"out_inc_{x}" for x in output_fields],
        #                             file_name=F"delete")
        id_field += 1


    return input_data, y_data

def generateXandY2D(input_fields_model, input_fields_obs, input_fields_var, output_field_increment,
                    field_names, obs_field_names, var_field_names, output_fields,
                    start_row, start_col, rows, cols, norm_type = PreprocParams.mean_var, perc_ocean = 1.0, uselatlon=False):
    """
    This function will generate X and Y boxes depening on the required field names and bboxes
    :param input_fields_model:
    :param input_fields_obs:
    :param output_field_increment:
    :param field_names:
    :param obs_field_names:
    :param output_fields:
    :param start_row:
    :param start_col:
    :param rows:
    :param cols:
    :return:
    """
    num_fields = len(field_names) + len(obs_field_names) + len(var_field_names)
    input_data = np.zeros((rows, cols, num_fields))
    tot_output_fields = len(output_fields)
    y_data = np.zeros((rows, cols, tot_output_fields))

    end_row = start_row + rows
    end_col = start_col + cols
    id_field = 0 # Id of the input fields

    coords_file = "/unity/g1/abozec/TSIS/GOMb0.04/topo/regional.grid.a"
    # coord_fields = [ 'plon','plat','qlon','qlat','ulon','ulat','vlon','vlat']
    coord_fields = ['plon','plat']

    hycom_coords = read_hycom_coords(coords_file, coord_fields)
    lats = hycom_coords['plat'][:,0]
    lons = hycom_coords['plon'][0,:]

    # Normalize the coordinates to be between 0 and 1
    lats = (lats - np.min(lats)) / (np.max(lats) - np.min(lats))
    lons = (lons - np.min(lons)) / (np.max(lons) - np.min(lons))

    lats = lats[start_row:end_row]
    lons = lons[start_col:end_col]

    # Make MESHGRID of the coordinates
    lats_mesh, lons_mesh = np.meshgrid(lats, lons)
    # Roll the latitude and longitude to be in the shape of the input data
    lats_mesh = np.rollaxis(lats_mesh, 0, 2)
    lons_mesh = np.rollaxis(lons_mesh, 0, 2)

    # ******* Ad![](../../../../../../../../../../data/HYCOM/DA_HYCOM_TSIS/Training/training_imgs/293_5_0_out.png)ding the model fields for input ********
    ssh_bias = 0
    for c_field in field_names:
        # This is just to try to incorporate the difference between obs and model into the input variables
        if c_field == "diff_ssh":
            modelssh = input_fields_model["srfhgt"][0, start_row:end_row, start_col:end_col]/9.806
            obsssh = input_fields_obs["ssh"][start_row:end_row, start_col:end_col].data.astype(np.float64)
            ssh_bias = np.nanmean(modelssh - obsssh)
            temp_data = obsssh - modelssh + ssh_bias
        elif c_field == "diff_sst":
            modelsst = input_fields_model["temp"][0, start_row:end_row, start_col:end_col]
            obssst = input_fields_obs["sst"][start_row:end_row, start_col:end_col].data.astype(np.float64)
            sst_bias = np.nanmean(modelsst - obssst)
            temp_data = obssst - modelsst + sst_bias
        elif c_field == "topo":
            # file_topo = "/data/HYCOM/DA_HYCOM_TSIS/Topography/gridinfo.nc"
            file_topo = "/unity/f1/ozavala/DATA/NN_HYCOM_TSIS/Topography/gridinfo.nc"
            topo_ds = xr.open_dataset(file_topo)
            depth_topo = topo_ds['mdepth']
            depth_mask = depth_topo > 100  # Depth we want to consider
            temp_data = depth_mask[start_row:end_row, start_col:end_col]
        else:
            temp_data = input_fields_model[c_field][0, start_row:end_row, start_col:end_col]
            c_perc_ocean = (temp_data.size - np.count_nonzero(np.isnan(temp_data)))/temp_data.size
            # We only validate ocean in non composite fields
            if c_perc_ocean < perc_ocean: # 99% ocean
                raise Exception(F"Not mostly ocean {c_field}")

        if c_field == "thknss":
            divide = 9806
            temp_data = temp_data/divide
        if c_field == "srfhgt":
            divide = 9.806
            temp_data = temp_data/divide

        # TODO couldn't find a better way to check that there are not 'land' pixels. IF YOU CHANGE IT YOU NEED TO BE SURE IT IS WORKING!!!!!!
        if c_field != "diff_sst" and c_field != "diff_ssh" and c_field != "topo":
            input_data[:, :, id_field] = normalizeData(temp_data, c_field, data_type=PreprocParams.type_model,
                                                    norm_type=norm_type, normalize=True)
        else:
            input_data[:, :, id_field] = temp_data

        if id_field == 0:
            c_mask = ma.getmaskarray(temp_data)
        id_field += 1
        # For debugging
        # import matplotlib.pyplot as plt
        # plt.imshow(temp_data.data)
        # plt.title(c_field)
        # plt.show()

    # ******* Adding the observations fields for input ********
    for c_field in obs_field_names:
        temp_data = input_fields_obs[c_field][start_row:end_row, start_col:end_col].data.astype(np.float64)
        if c_field == "ssh":
            temp_data += ssh_bias
        input_data[:, :, id_field] = normalizeData(temp_data, c_field, data_type=PreprocParams.type_obs,
                                                   norm_type=norm_type, normalize=True)
        # input_data[:, :, id_field] = temp_data
        id_field += 1

    # ******* Adding the variance fields for input ********
    if len(var_field_names) > 0:
        print("Adding variance fields...")
        for c_field in var_field_names:
            input_data[:, :, id_field] = normalizeData(temp_data, c_field, data_type=PreprocParams.type_obs,
                                                   norm_type=norm_type, normalize=True)
            # temp_data = input_fields_var[c_field][0, start_row:end_row, start_col:end_col]
            input_data[:, :, id_field] = temp_data
            id_field += 1

    # ******************* Filling the 'output' data ***************
    id_field = 0
    for c_field in output_fields:
        back_data = input_fields_model[c_field][0, start_row:end_row, start_col:end_col]
        if c_field == "thknss":
            divide = 9806
            back_data = back_data/divide
        if c_field == "srfhgt":
            divide = 9.806
            back_data = back_data/divide

        # Here we compute the increment as TSIS - Model
        temp_data = output_field_increment[c_field][0, start_row:end_row, start_col:end_col] - back_data
        # TODO this is a hack, for some reason the output field is not properly masked. So we force it to nan
        temp_data[c_mask] = np.nan
        y_data[:, :, id_field] = normalizeData(temp_data, c_field, data_type=PreprocParams.type_inc,
                                                   norm_type=norm_type, normalize=True)
        id_field += 1

    # Add the coordinates to the input data
    if uselatlon:
        input_data = np.dstack((input_data, lats_mesh, lons_mesh))

    return input_data, y_data

def generateXandY3D(input_fields_model, input_fields_obs, input_fields_var, output_field_increment,
                  field_names, obs_field_names, var_field_names, output_fields,
                  start_row, start_col, rows, cols, z_layers, norm_type = PreprocParams.mean_var, perc_ocean = 1.0):
    """
    This function will generate X and Y boxes depening on the required field names and bboxes
    :param input_fields_model:
    :param input_fields_obs:
    :param output_field_increment:
    :param field_names:
    :param obs_field_names:
    :param output_fields:
    :param start_row:
    :param start_col:
    :param rows:
    :param cols:
    :return:
    """
    num_fields = len(field_names) + len(obs_field_names) + len(var_field_names)
    input_data = np.zeros((rows, cols, len(z_layers), num_fields))
    tot_output_fields = len(output_fields)
    y_data = np.zeros((rows, cols, len(z_layers), tot_output_fields))

    end_row = start_row + rows
    end_col = start_col + cols
    id_field = 0 # Id of the input fields

    # ******* Adding the model fields for input ********
    for c_field in field_names:
        # Verify for 2D fields. If only 2D then just copy the surface data
        # input_data[:, :, id_field] = normalizeData(temp_data, c_field, data_type=PreprocParams.type_model,
        #                                            norm_type=norm_type, normalize=True)
        if input_fields_model[c_field].shape[0] > 1:
            temp_data = input_fields_model[c_field][z_layers, start_row:end_row, start_col:end_col]
            input_data[:, :, :, id_field] = np.rollaxis(temp_data, 0, 3)
        else:
            temp_data = input_fields_model[c_field][0, start_row:end_row, start_col:end_col]
            input_data[:, :, 0, id_field] = temp_data
        id_field += 1
    # import matplotlib.pyplot as plt
    # plt.imshow(input_data[:,:,0, 2])
    # plt.show()

    # ******* Adding the observations fields for input ********
    for c_field in obs_field_names:
        # We assume all observations are at most in the surface (no 3D)
        temp_data = input_fields_obs[c_field][start_row:end_row, start_col:end_col].astype(np.float64)
        mask = temp_data.mask
        temp_data[mask] = np.nan
        input_data[:, :, 0, id_field] = temp_data
        id_field += 1

    # ******* Adding the variance fields for input ********
    for c_field in var_field_names:
        # TODO z_layers are hardcoded, should have been filtered before
        if len(input_fields_var[c_field].shape) > 2:
            temp_data = input_fields_var[c_field][0, start_row:end_row, start_col:end_col, 0:z_layers]
        else:
            temp_data = input_fields_var[c_field][start_row:end_row, start_col:end_col, 0:z_layers]
        # input_data[:, :, id_field] = normalizeData(temp_data, c_field, data_type=PreprocParams.type_obs,
        #                                            norm_type=norm_type, normalize=True)

        input_data[:, :, id_field] = temp_data
        id_field += 1

    # ******************* Filling the 'output' data ***************
    id_field = 0
    for c_field in output_fields:
        # Verify for 2D fields. If only 2D then just copy the surface data
        if c_field == "thknss":
            divide = 9806
        temp_data = temp_data/divide
        if c_field == "srfhgt":
            divide = 9.806
            temp_data = temp_data/divide
        if input_fields_model[c_field].shape[0] > 1:
            temp_data = output_field_increment[c_field][z_layers, start_row:end_row, start_col:end_col]
            y_data[:, :, :, id_field] = np.rollaxis(temp_data, 0, 3)
        else:
            temp_data = output_field_increment[c_field][0, start_row:end_row, start_col:end_col]
            y_data[:, :, 0, id_field] = temp_data
        id_field += 1

    return input_data, y_data


def get_date_from_preproc_filename(file_name):
    split_name = file_name.split('/')[-1].split('_')
    year = int(split_name[1])
    day = int(split_name[2].split('.')[0])
    return year, day
