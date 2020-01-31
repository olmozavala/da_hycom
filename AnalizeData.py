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
from config.MainConfig import get_analize_data_config, get_parallel_config
from preproc.UtilsDates import get_day_of_year_from_month_and_day
from sklearn.metrics import mean_squared_error

# Not sure how to move this inside the function
config_par = get_parallel_config()
NUM_PROC = config_par[ParallelParams.NUM_PROC]

def parallel_proc(proc_id):
    # /data/COAPS_nexsan/people/abozec/TSIS/IASx0.03/obs/qcobs_mdt_gofs/WITH_PIES
    config = get_analize_data_config()
    input_folder_tsis = config[PreprocParams.input_folder_tsis]
    input_folder_forecast = config[PreprocParams.input_folder_forecast]
    input_folder_obs = config[PreprocParams.input_folder_obs]
    output_folder = config[PreprocParams.output_folder]
    PIES = config[PreprocParams.PIES]
    YEARS = config[PreprocParams.YEARS]
    MONTHS = config[PreprocParams.MONTHS]
    fields = config[PreprocParams.fields_names]
    fields_obs = config[PreprocParams.fields_names_obs]
    plot_modes = config[PreprocParams.plot_modes_per_field]
    layers = config[PreprocParams.layers_to_plot]

    img_viz = EOAImageVisualizer(output_folder=output_folder, disp_images=False)

    # These are the data assimilated files
    for c_using_pie in PIES:
        for c_year in YEARS:
            for c_month in MONTHS:
                try:
                    days_of_month, days_of_year = get_days_from_month(c_month)
                    # Rads all the files for this month
                    da_files, da_paths = get_da_file_name(input_folder_tsis, c_year, c_month, c_using_pie)
                    forecast_files, forecast_paths = get_forecast_file_name(input_folder_forecast, c_year,
                                                                                        c_month, c_using_pie)
                    obs_files, obs_paths = get_obs_file_names(input_folder_obs, c_year, c_month, c_using_pie)
                except Exception as e:
                    print(F"Failed to find all files for date {c_year}-{c_month}")

                # This for is fixed to be able to run in parallel
                for c_day_of_month, c_day_of_year in enumerate(days_of_year):
                    if (c_day_of_month % NUM_PROC) == proc_id:
                        re_hycom = F'archv.{c_year}_{c_day_of_year:03d}\S*.a'
                        re_obs = F'tsis_obs_ias_{c_year}{c_month:02d}{c_day_of_month+1:02d}\S*.nc'

                        try:
                            da_file_idx = [i for i, file in enumerate(da_files) if re.search(re_hycom, file) != None][0]
                            forecast_file_idx = [i for i, file in enumerate(forecast_files)
                                                        if re.search(re_hycom, file) != None][0]
                            obs_file_idx = [i for i, file in enumerate(obs_files)
                                                       if re.search(re_obs, file) != None][0]

                        except Exception as e:
                            print(F"ERROR: The file for date {c_year} - {c_month} - {c_day_of_month} doesn't exist: {e}")
                            continue

                        print(F" =============== Working with: {da_files[da_file_idx]} ============= ")
                        print(F"Available fields: {read_field_names(da_paths[da_file_idx])}")
                        da_np_fields = read_hycom_output(da_paths[da_file_idx], fields, layers=layers)
                        forecast_np_fields = read_hycom_output(forecast_paths[forecast_file_idx], fields, layers=layers)
                        obs_np_fields = read_netcdf(obs_paths[obs_file_idx], fields_obs, layers=[0], rename_fields=fields)

                        for idx_field, c_field_name in enumerate(fields):
                            da_np_c_field = da_np_fields[c_field_name]
                            forecast_np_c_field = forecast_np_fields[c_field_name]
                            # TODO patch for ssh
                            if c_field_name == 'srfhgt':
                                obs_np_c_field = np.array(obs_np_fields[c_field_name][:].filled(np.nan))*10
                            else:
                                obs_np_c_field = obs_np_fields[c_field_name]

                            diff = da_np_c_field - forecast_np_c_field
                            # In these 2 cases, we only compute it for the surface layer
                            diff_obs_vs_hycom = obs_np_c_field - forecast_np_c_field[0]
                            obs_np_c_field[502,609] - forecast_np_c_field[0][502,609]
                            diff_obs_vs_da = obs_np_c_field - da_np_c_field[0]

                            mse_hycom_vs_da = mse(da_np_c_field, forecast_np_c_field)
                            mse_obs_vs_hycom = mse(obs_np_c_field, forecast_np_c_field[0])
                            mse_obs_vs_da = mse(obs_np_c_field, da_np_c_field[0])

                            title = F"{c_field_name} {c_year}_{c_month:02d}_{c_day_of_month:02d}"
                            img_viz.plot_3d_data_np([np.expand_dims(obs_np_c_field, 0), forecast_np_c_field, da_np_c_field,
                                                     diff, np.expand_dims(diff_obs_vs_hycom, 0), np.expand_dims(diff_obs_vs_da, 0)],
                                                    var_names=['Obs', 'HYCOM', 'DA', F'DA-Forecast (mse:{mse_hycom_vs_da:.3f})',
                                                               F'Obs-Hycom (mse:{mse_obs_vs_hycom:.3f})',
                                                               F'Obs-DA (mse:{mse_obs_vs_da:.3f})'],
                                                    title=title, file_name_prefix=F'{c_field_name}_{c_year}_{c_month:02d}_{c_day_of_month:02d}', z_lavels_names=layers,
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
                    da_file = F'/data/COAPS_Net/gleam/dmitry/hycom/TSIS/IASx0.03/output/{year}_PIES/archv.{year}_{day_of_year}_00.a'

                    freerun_data = read_hycom_output(freerun_file, fields, [0])
                    da_data = read_hycom_output(da_file, fields, [0])
                    not_nan_idx = np.logical_not(np.isnan(np.array(freerun_data[field][0])))

                    fig, axs = plt.subplots(1, 3, squeeze=True, figsize=(16 * 3, 16))
                    axs[0].imshow(freerun_data[field][0])
                    axs[0].set_title(F"Freerun {field}", fontdict={'fontsize': 80})
                    axs[1].imshow(da_data[field][0])
                    axs[1].set_title(F"DA {field}", fontdict={'fontsize': 80})
                    axs[2].imshow(freerun_data[field][0] - da_data[field][0])
                    axs[2].set_title(
                        F"Difference MSE ~{mean_squared_error(np.array(da_data[field][0])[not_nan_idx], np.array(freerun_data[field][0])[not_nan_idx]):0.4f}",
                        fontdict={'fontsize': 80})
                    fig.suptitle(F"{year}_{month}_{field}", fontsize=80)
                    plt.show()
                except Exception as e:
                    print(F"Failed for {year}_{month}_{field}: {e}")
                    continue


def main():
    # ----------- Parallel -------
    # p = Pool(NUM_PROC)
    # p.map(parallel_proc, range(NUM_PROC))

    # ----------- Sequencial -------
    NUM_PROC = 1
    # parallel_proc(1)
    testing()


if __name__ == '__main__':
    main()
