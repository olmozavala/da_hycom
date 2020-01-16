from preproc.UtilsDates import get_days_from_month
from os.path import join
import os
import numpy as np

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

def get_da_file_name(input_folder, year, month, pies):
    """
    This function obtains the complete path of the files for the specified month and year, stored by Dmitry with DA
    :param input_folder:
    :param year:
    :param month:
    :param pies:
    :return: str array [file names], str array [file paths]
    """
    pies_txt = 'PIES' if pies else 'noPIES'

    _, days_of_year = get_days_from_month(month)
    folder = join(input_folder, F'{str(year)}_{pies_txt}')
    all_files = os.listdir(folder)
    selected_files = [file for file in all_files if int(file.split('_')[1]) in days_of_year]
    selected_files.sort()
    return selected_files, [join(folder,c_file) for c_file in selected_files]


def get_forecast_file_name(input_folder, year, month, pies):
    """
    This function obtains the complete path of the files for the specified month and year, stored by Alec
    :param input_folder:
    :param year:
    :param month:
    :param pies:
    :return:
    """
    pies_txt = 'PIES' if pies else 'noPIES'

    folder = join(input_folder, F'{pies_txt}',F'{year}{month:02d}')
    all_files = os.listdir(folder)
    all_files.sort()
    return all_files, [join(folder,c_file) for c_file in all_files]


def get_obs_file_names(input_folder, year, month, pies=True):
    """
    This function obtains the complete path of the files for the specified month and year, observations
    :param input_folder:
    :param year:
    :param month:
    :param pies:
    :return:
    """
    pies_txt = 'WITH_PIES' if pies else 'WITHOUT_PIES'

    folder = join(input_folder, F'{pies_txt}')
    all_files = os.listdir(folder)
    selected_files = [file for file in all_files if (file.split('_')[3][0:6]) == F'{year}{month:02d}']
    selected_files.sort()
    return selected_files, [join(folder, c_file) for c_file in selected_files]