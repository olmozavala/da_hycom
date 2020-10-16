from info.info_hycom import read_field_names
from inout.io_hycom import read_hycom_output
from inout.io_netcdf import read_netcdf
from img_viz.eoa_viz import EOAImageVisualizer
from img_viz.constants import PlotMode
from preproc.UtilsDates import get_days_from_month
from preproc.metrics import mse
from io_project.read_utils import *
import re
import numpy as np
from multiprocessing import Pool
from constants_proj.AI_proj_params import PreprocParams, ParallelParams
from config.PreprocConfig import get_preproc_config
from preproc.UtilsDates import get_day_of_year_from_month_and_day
from sklearn.metrics import mean_squared_error

NUM_PROC = 1

def main():
    # ----------- Parallel -------
    p = Pool(NUM_PROC)
    # compute_consecutive_days_difference()
    p.map(img_generation_all, range(NUM_PROC))
    # p.map(img_generation_hycom, range(NUM_PROC))
    # img_generation_all(1)  # Single proc generation

    # ----------- Sequencial -------
    # img_generation_all(1)
    # testing()

def img_generation_all(proc_id):
    """
    Makes images of the available data (Free run, DA and Observations)
    :param proc_id:
    :return:
    """
    config = get_preproc_config()
    input_folder_tsis = config[PreprocParams.input_folder_tsis]
    input_folder_forecast = config[PreprocParams.input_folder_hycom]
    input_folder_obs = config[PreprocParams.input_folder_obs]
    output_folder = config[PreprocParams.imgs_output_folder]
    YEARS = config[PreprocParams.YEARS]
    MONTHS = config[PreprocParams.MONTHS]
    fields = config[PreprocParams.fields_names]
    fields_obs = config[PreprocParams.fields_names_obs]
    plot_modes = config[PreprocParams.plot_modes_per_field]
    layers = config[PreprocParams.layers_to_plot]

    img_viz = EOAImageVisualizer(output_folder=output_folder, disp_images=False)

    # Iterate current year
    for c_year in YEARS:
        # Iterate current month
        for c_month in MONTHS:
            try:
                days_of_month, days_of_year = get_days_from_month(c_month)
                # Reads the data (DA, Free run, and observations)
                increment_files, increment_paths = get_hycom_file_name(input_folder_tsis, c_year, c_month)
                hycom_files, hycom_paths = get_hycom_file_name(input_folder_forecast, c_year, c_month)
                obs_files, obs_paths = get_obs_file_names(input_folder_obs, c_year, c_month)
            except Exception as e:
                print(F"Failed to find any file for date {c_year}-{c_month}")
                continue

            # This for is fixed to be able to run in parallel
            for c_day_of_month, c_day_of_year in enumerate(days_of_year):
                if (c_day_of_month % NUM_PROC) == proc_id:
                    # Makes regular expression of the current desired file
                    re_tsis = F'incupd.{c_year}_{c_day_of_year:03d}\S*.a'
                    re_hycom = F'archv.{c_year}_{c_day_of_year:03d}\S*.a'
                    re_obs = F'tsis_obs_ias_{c_year}{c_month:02d}{c_day_of_month+1:02d}\S*.nc'

                    try:
                        # Gets the proper index of the file for the three cases
                        increment_file_idx = [i for i, file in enumerate(increment_files) if re.search(re_tsis, file) != None][0]
                        hycom_file_idx = [i for i, file in enumerate(hycom_files) if re.search(re_hycom, file) != None][0]
                        obs_file_idx = [i for i, file in enumerate(obs_files) if re.search(re_obs, file) != None][0]
                    except Exception as e:
                        print(F"ERROR: The file for date {c_year} - {c_month} - {(c_day_of_month+1)} doesn't exist: {e}")
                        continue

                    print(F" =============== Working with: {increment_files[increment_file_idx]} ============= ")
                    print(F"Available fields on increment: {read_field_names(increment_paths[increment_file_idx])}")
                    increment_np_fields = read_hycom_output(increment_paths[increment_file_idx], fields, layers=layers)
                    model_state_np_fields = read_hycom_output(hycom_paths[hycom_file_idx], fields, layers=layers)
                    obs_np_fields = read_netcdf(obs_paths[obs_file_idx], fields_obs, layers=[0], rename_fields=fields)

                    for idx_field, c_field_name in enumerate(fields):
                        increment_np_c_field = increment_np_fields[c_field_name]
                        nan_indx = increment_np_c_field == 0
                        increment_np_c_field[nan_indx] = np.nan
                        model_state_np_c_field = model_state_np_fields[c_field_name]
                        obs_np_c_field = obs_np_fields[c_field_name]

                        # diff_increment_vs_fo = increment_np_c_field - model_state_np_c_field
                        # In these 2 cases, we only compute it for the surface layer
                        # diff_obs_vs_hycom = obs_np_c_field - model_state_np_c_field[0]
                        obs_np_c_field[502,609] - model_state_np_c_field[0][502,609]
                        # diff_obs_vs_da = obs_np_c_field - increment_np_c_field[0]

                        # mse_hycom_vs_da = mse(increment_np_c_field, model_state_np_c_field)
                        # mse_obs_vs_hycom = mse(obs_np_c_field, model_state_np_c_field[0])
                        # mse_obs_vs_da = mse(obs_np_c_field, increment_np_c_field[0])

                        title = F"{c_field_name} {c_year}_{c_month:02d}_{(c_day_of_month+1):02d}"
                        # ======================= Only Fredatae HYCOM, TSIS, Observations ==================
                        img_viz.plot_3d_data_np([np.expand_dims(obs_np_c_field, 0), model_state_np_c_field, increment_np_c_field],
                                                var_names=[F'Observations', 'HYCOM', 'Increment (TSIS)'],
                                                title=title, file_name_prefix=F'Summary_{c_field_name}_{c_year}_{c_month:02d}_{(c_day_of_month+1):02d}', z_lavels_names=layers,
                                                flip_data=True, plot_mode=plot_modes[idx_field])

                        # img_viz.plot_3d_data_np([np.expand_dims(obs_np_c_field, 0), model_state_np_c_field, increment_np_c_field,
                        #                          diff_increment_vs_fo, np.expand_dims(diff_obs_vs_hycom, 0), np.expand_dims(diff_obs_vs_da, 0)],
                        #                         var_names=['Obs', 'HYCOM', 'DA', F'DA-Forecast (mse:{mse_hycom_vs_da:.3f})',
                        #                                    F'Obs-Hycom (mse:{mse_obs_vs_hycom:.3f})',
                        #                                    F'Obs-DA (mse:{mse_obs_vs_da:.3f})'],
                        #                         title=title, file_name_prefix=F'{c_field_name}_{c_year}_{c_month:02d}_{c_day_of_month:02d}', z_lavels_names=layers,
                        #                         flip_data=True, plot_mode=plot_modes[idx_field])

                        # ======================= Only Free HYCOM, TSIS assimilated and the difference ==================
                        # img_viz.plot_3d_data_np([model_state_np_c_field, increment_np_c_field, diff_increment_vs_fo],
                        #                         var_names=['Free', 'TSIS', F'TSIS-Free'],
                        #                         title=title, file_name_prefix=F'0_{c_field_name}_{c_year}_{c_month:02d}_{c_day_of_month:02d}', z_lavels_names=layers,
                        #                         flip_data=True, plot_mode=plot_modes[idx_field])

def compute_consecutive_days_difference():
    """
    Computes the difference between consecutive days on the hycom files.
    :param proc_id:
    :return:
    """
    config = get_preproc_config()
    input_folder_forecast = config[PreprocParams.input_folder_hycom]
    output_folder = config[PreprocParams.imgs_output_folder]
    YEARS = config[PreprocParams.YEARS]
    MONTHS = config[PreprocParams.MONTHS]
    fields = config[PreprocParams.fields_names]
    layers = config[PreprocParams.layers_to_plot]

    img_viz = EOAImageVisualizer(output_folder=output_folder, disp_images=False)

    # Iterate current year
    for c_year in YEARS:
        # Iterate current month
        diff_per_field = {field:[] for field in fields}
        days_with_data = []
        for c_month in MONTHS:
            # Reading the data
            try:
                days_of_month, days_of_year = get_days_from_month(c_month)
                # Reading hycom files
                hycom_files, hycom_paths = get_hycom_file_name(input_folder_forecast, c_year, c_month)
            except Exception as e:
                print(F"Failed to find any file for date {c_year}-{c_month}")
                continue

            # This for is fixed to be able to run in parallel
            for c_day_of_month, c_day_of_year in enumerate(days_of_year):
                print(F"---------- Year {c_year} day: {c_day_of_year} --------------")
                # Makes regular expression of the current desired file
                re_hycom = F'archv.{c_year}_{c_day_of_year:03d}\S*.a'
                re_hycom_prev = F'archv.{c_year}_{(c_day_of_year-1):03d}\S*.a'
                try:
                    # Gets the proper index of the file for the three cases
                    hycom_file_idx = [i for i, file in enumerate(hycom_files) if re.search(re_hycom, file) != None][0]
                    hycom_file_idx_prev = [i for i, file in enumerate(hycom_files) if re.search(re_hycom_prev, file) != None][0]
                except Exception as e:
                    print(F"ERROR: The file for date {c_year} - {c_month} - {c_day_of_month} (and prev day) don't exist: {e}")
                    continue

                days_with_data.append(c_day_of_year)
                model_state_np_fields = read_hycom_output(hycom_paths[hycom_file_idx], fields, layers=layers)
                model_state_np_fields_prev = read_hycom_output(hycom_paths[hycom_file_idx_prev], fields, layers=layers)
                # Computes the difference between consecutive days from the desired fields
                for idx_field, c_field_name in enumerate(fields):
                    model_state_np_c_field = model_state_np_fields[c_field_name]
                    model_state_np_c_field_prev = model_state_np_fields_prev[c_field_name]
                    c_diff = np.abs(np.nanmean(model_state_np_c_field_prev - model_state_np_c_field))
                    diff_per_field[c_field_name].append(c_diff)

        # Plots the differences between consecutive days. For all the fields together.
        img_viz.plot_1d_data_np(days_with_data, [diff_per_field[a] for a in diff_per_field.keys()],  title='Difference between days',
                                labels=fields, file_name_prefix='HYCOM_Diff_Between_Days',
                                wide_ratio=4)
        # Plots the differences between consecutive days. Separated by fields
        for field in diff_per_field.keys():
            img_viz.plot_1d_data_np(days_with_data, [diff_per_field[field]],  title=F'Difference between days {field}',
                                    labels=[field], file_name_prefix=F'HYCOM_Diff_Between_Days_{field}',
                                    wide_ratio=4)

def img_generation_hycom(proc_id):
    """
    Makes images of the available data (Free run, DA and Observations)
    :param proc_id:
    :return:
    """
    config = get_preproc_config()
    input_folder_tsis = config[PreprocParams.input_folder_tsis]
    input_folder_forecast = config[PreprocParams.input_folder_hycom]
    input_folder_obs = config[PreprocParams.input_folder_obs]
    output_folder = config[PreprocParams.imgs_output_folder]
    YEARS = config[PreprocParams.YEARS]
    MONTHS = config[PreprocParams.MONTHS]
    fields = config[PreprocParams.fields_names]
    fields_obs = config[PreprocParams.fields_names_obs]
    plot_modes = config[PreprocParams.plot_modes_per_field]
    layers = config[PreprocParams.layers_to_plot]

    img_viz = EOAImageVisualizer(output_folder=output_folder, disp_images=False)

    # Iterate current year
    for c_year in YEARS:
        # Iterate current month
        for c_month in MONTHS:
            try:
                days_of_month, days_of_year = get_days_from_month(c_month)
                # Reads the data (DA, Free run, and observations)
                hycom_files, hycom_paths = get_hycom_file_name(input_folder_forecast, c_year, c_month)
            except Exception as e:
                print(F"Failed to find any file for date {c_year}-{c_month}")
                continue

            # This for is fixed to be able to run in parallel
            for c_day_of_month, c_day_of_year in enumerate(days_of_year):
                if (c_day_of_month % NUM_PROC) == proc_id:
                    # Makes regular expression of the current desired file
                    re_hycom = F'archv.{c_year}_{c_day_of_year:03d}\S*.a'
                    try:
                        # Gets the proper index of the file for the three cases
                        hycom_file_idx = [i for i, file in enumerate(hycom_files) if re.search(re_hycom, file) != None][0]
                    except Exception as e:
                        print(F"ERROR: The file for date {c_year} - {c_month} - {c_day_of_month} doesn't exist: {e}")
                        continue

                    print(F" =============== Working with: {hycom_files[hycom_file_idx]} ============= ")
                    print(F"Available fields: {read_field_names(hycom_paths[hycom_file_idx])}")
                    model_state_np_fields = read_hycom_output(hycom_paths[hycom_file_idx], fields, layers=layers)
                    for idx_field, c_field_name in enumerate(fields):
                        model_state_np_c_field = model_state_np_fields[c_field_name]
                        title = F"{c_field_name} {c_year}_{c_month:02d}_{(c_day_of_month+1):02d}"
                        # ======================= Only Fredatae HYCOM, TSIS, Observations ==================
                        img_viz.plot_3d_data_np([model_state_np_c_field],
                                                var_names=[F'HYCOM'],
                                                title=title, file_name_prefix=F'HYCOM_{c_field_name}_{c_year}_{c_month:02d}_{c_day_of_month:02d}', z_lavels_names=layers,
                                                flip_data=True, plot_mode=plot_modes[idx_field])


def testing():
    import matplotlib.pyplot as plt
    import math

    # Computing the MSE between the 'free run and the DA surn
    input_folder = '/data/HYCOM/DA_HYCOM_TSIS/preproc'
    years = [2009, 2010, 2011]
    months = range(1,13)
    day_month = 2
    fields = ['srfhgt']
    for year in years:
        for month in months:
            for field in fields:
                try:
                    day_of_year = get_day_of_year_from_month_and_day(month, day_of_month=day_month, year=year)
                    freerun_file = F'/data/COAPS_Net/gleam/abozec/HYCOM/TSIS/IASx0.03/forecast/PIES/{year}{month:02d}/archv.{year}_{day_of_year:03d}_00.a'
                    increment_file = F'/data/COAPS_Net/gleam/dmitry/hycom/TSIS/IASx0.03/output/{year}_PIES/archv.{year}_{day_of_year}_00.a'

                    freerun_data = read_hycom_output(freerun_file, fields, [0])
                    increment_data = read_hycom_output(increment_file, fields, [0])
                    not_nan_idx = np.logical_not(np.isnan(np.array(freerun_data[field][0])))

                    fig, axs = plt.subplots(1, 3, squeeze=True, figsize=(16 * 3, 16))
                    axs[0].imshow(freerun_data[field][0])
                    axs[0].set_title(F"Freerun {field}", fontdict={'fontsize': 80})
                    axs[1].imshow(increment_data[field][0])
                    axs[1].set_title(F"DA {field}", fontdict={'fontsize': 80})
                    axs[2].imshow(freerun_data[field][0] - increment_data[field][0])
                    axs[2].set_title(
                        F"Difference MSE ~{mean_squared_error(np.array(increment_data[field][0])[not_nan_idx], np.array(freerun_data[field][0])[not_nan_idx]):0.4f}",
                        fontdict={'fontsize': 80})
                    fig.suptitle(F"{year}_{month}_{field}", fontsize=80)
                    plt.show()
                except Exception as e:
                    print(F"Failed for {year}_{month}_{field}: {e}")
                    continue


if __name__ == '__main__':
    main()
