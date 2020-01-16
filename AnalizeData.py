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
from constants_proj.AI_proj_params import AnalizeDataParams, ParallelParams
from config.MainConfig import get_analize_data_config, get_paralallel_config

# Not sure how to move this inside the function
config_par = get_paralallel_config()
NUM_PROC = config_par[ParallelParams.NUM_PROC]

def parallel_proc(proc_id):
    # /data/COAPS_nexsan/people/abozec/TSIS/IASx0.03/obs/qcobs_mdt_gofs/WITH_PIES
    config = get_analize_data_config()
    input_folder_tsis = config[AnalizeDataParams.input_folder_tsis]
    input_folder_forecast = config[AnalizeDataParams.input_folder_forecast]
    input_folder_obs = config[AnalizeDataParams.input_folder_obs]
    output_folder = config[AnalizeDataParams.output_folder]
    PIES = config[AnalizeDataParams.PIES]
    YEARS = config[AnalizeDataParams.YEARS]
    MONTHS = config[AnalizeDataParams.MONTHS]
    fields = config[AnalizeDataParams.fields_names]
    fields_obs = config[AnalizeDataParams.fields_names_obs]
    plot_modes = config[AnalizeDataParams.plot_modes_per_field]
    layers = config[AnalizeDataParams.layers_to_plot]

    img_viz = EOAImageVisualizer(output_folder=output_folder, disp_images=False)

    # These are the data assimilated files
    for c_using_pie in PIES:
        for c_year in YEARS:
            for c_month in MONTHS:
                days_of_month, days_of_year = get_days_from_month(c_month)
                # Rads all the files for this month
                da_files, da_paths = get_da_file_name(input_folder_tsis, c_year, c_month, c_using_pie)
                forecast_files, forecast_paths = get_forecast_file_name(input_folder_forecast, c_year,
                                                                                    c_month, c_using_pie)
                obs_files, obs_paths = get_obs_file_names(input_folder_obs, c_year, c_month, c_using_pie)

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


def main():
    # ----------- Parallel -------
    # p = Pool(NUM_PROC)
    # p.map(parallel_proc, range(NUM_PROC))

    # ----------- Sequencial -------
    NUM_PROC = 1
    parallel_proc(1)

if __name__ == '__main__':
    main()
