from preproc.UtilsDates import get_days_from_month
import numpy.ma as ma
from constants_proj.AI_proj_params import MAX_MODEL, MAX_OBS, MIN_OBS, MIN_MODEL, MAX_INCREMENT, MIN_INCREMENT
from os.path import join, isfile
import os
import numpy as np

def get_preproc_increment_files(input_folder):
    all_files = os.listdir(input_folder)
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

def get_hycom_file_name(input_folder, year, month):
    """
    This function obtains the complete path of the files for the specified month and year, stored by Dmitry with DA
    :param input_folder:
    :param year:
    :param month:
    :param pies:
    :return: str array [file names], str array [file paths]
    """
    _, days_of_year = get_days_from_month(month)
    # folder = join(input_folder, F'{str(year)}')
    folder = input_folder
    all_files = os.listdir(folder)
    all_files = [x for x in all_files  if isfile(join(input_folder, x))]
    selected_files = [file for file in all_files if int(file.split('_')[1]) in days_of_year]
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
    all_files = [file for file in all_files if file.find('tsis_obs_ias') != -1] # Avoid additiona files
    selected_files = [file for file in all_files if (file.split('_')[3][0:6]) == F'{year}{month:02d}']
    selected_files.sort()
    return selected_files, [join(folder, c_file) for c_file in selected_files]

def generateXandY(input_fields_model, input_fields_obs, output_field_increment, field_names, obs_field_names, output_fields,
                  start_row, start_col, rows, cols):
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
    num_fields = len(field_names) + len(obs_field_names)
    input_data = np.zeros((rows, cols, num_fields))
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
        if len(temp_data.mask.shape) != 1: # It means it has at least tome ocean values
            if temp_data.mask.min() == False or np.dtype('bool') != temp_data.mask.min(): # Making sure there is at leas some ocean pixels
                input_data[:, :, id_field] = (temp_data - MIN_MODEL[c_field]) /(MAX_MODEL[c_field] - MIN_MODEL[c_field])
                if id_field == 0:
                    c_mask = ma.getmaskarray(temp_data)
                id_field += 1
            else:
                raise Exception("Only land")
        else:
            raise Exception("Only land")

    # ******* Adding the observations fields for input ********
    for c_field in obs_field_names:
        temp_data = input_fields_obs[c_field][start_row:end_row, start_col:end_col]
        input_data[:, :, id_field] = (temp_data.filled(np.nan) - MIN_OBS[c_field])/(MAX_OBS[c_field] - MIN_OBS[c_field])
        id_field += 1

    # ******************* Filling the 'output' data ***************
    id_field = 0
    for c_field in output_fields:
        temp_data = output_field_increment[c_field][start_row:end_row, start_col:end_col]
        # TODO this is a hack, for some reason the output field is not properly masked. So we force it to nan
        temp_data[c_mask] = np.nan
        y_data[:, :, id_field] = (temp_data - MIN_INCREMENT[c_field]) /(MAX_INCREMENT[c_field] - MIN_INCREMENT[c_field])
        # viz_obj = EOAImageVisualizer(output_folder=join(input_folder_preproc, "training_imgs"), disp_images=True)
        # viz_obj.plot_2d_data_np_raw(y_data.swapaxes(0,2),
        #                             var_names= [F"out_inc_{x}" for x in output_fields],
        #                             file_name=F"delete")
        id_field += 1

    return input_data, y_data
    # return input_data, input_data

def get_date_from_preproc_filename(file_name):
    split_name = file_name.split('/')[-1].split('_')
    year = int(split_name[1])
    day = int(split_name[2].split('.')[0])
    return year, day
