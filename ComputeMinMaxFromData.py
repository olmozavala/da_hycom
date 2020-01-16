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
    input_folder_obs = config[AnalizeDataParams.input_folder_obs]
    output_folder = config[AnalizeDataParams.output_folder]
    PIES = config[AnalizeDataParams.PIES]
    YEARS = config[AnalizeDataParams.YEARS]
    MONTHS = config[AnalizeDataParams.MONTHS]
    fields = config[AnalizeDataParams.fields_names]
    fields_obs = config[AnalizeDataParams.fields_names_obs]
    layers = config[AnalizeDataParams.layers_to_plot]

    max_values = {field: 0 for field in fields}
    min_values = {field: 10**5 for field in fields}
    max_values_obs = {field: 0 for field in fields_obs}
    min_values_obs = {field: 10**5 for field in fields_obs}

    # These are the data assimilated files
    for c_using_pie in PIES:
        for c_year in YEARS:
            for c_month in MONTHS:
                days_of_month, days_of_year = get_days_from_month(c_month)
                # Rads all the files for this month
                da_files, da_paths = get_da_file_name(input_folder_tsis, c_year, c_month, c_using_pie)
                obs_files, obs_paths = get_obs_file_names(input_folder_obs, c_year, c_month, c_using_pie)

                # This for is fixed to be able to run in parallel
                for c_day_of_month, c_day_of_year in enumerate(days_of_year):
                    if (c_day_of_month % NUM_PROC) == proc_id:
                        re_hycom = F'archv.{c_year}_{c_day_of_year:03d}\S*.a'
                        re_obs = F'tsis_obs_ias_{c_year}{c_month:02d}{c_day_of_month+1:02d}\S*.nc'

                        try:
                            da_file_idx = [i for i, file in enumerate(da_files) if re.search(re_hycom, file) != None][0]
                            obs_file_idx = [i for i, file in enumerate(obs_files) if re.search(re_obs, file) != None][0]
                        except Exception as e:
                            print(F"ERROR: The file for date {c_year} - {c_month} - {c_day_of_month} doesn't exist: {e}")
                            continue

                        print(F" =============== Working with: {da_files[da_file_idx]} ============= ")
                        print(F"Available fields: {read_field_names(da_paths[da_file_idx])}")
                        # da_np_fields = read_hycom_output(da_paths[da_file_idx], fields, layers=layers)
                        #
                        # for idx_field, c_field_name in enumerate(fields):
                        #     da_np_c_field = da_np_fields[c_field_name]
                        #     c_max = np.nanmax(da_np_c_field)
                        #     c_min = np.nanmin(da_np_c_field)
                        #     if c_max >= max_values[c_field_name]:
                        #         max_values[c_field_name] = c_max
                        #     if c_min <= min_values[c_field_name]:
                        #         min_values[c_field_name] = c_min

                        obs_np_fields = read_netcdf(obs_paths[obs_file_idx], fields_obs, layers=[0], rename_fields=fields)

                        for idx_field, c_field_name in enumerate(fields):
                            obs_np_c_field = obs_np_fields[c_field_name]
                            c_max = np.nanmax(obs_np_c_field)
                            c_min = np.nanmin(obs_np_c_field)
                            if c_max >= max_values_obs[c_field_name]:
                                max_values_obs[c_field_name] = c_max
                            if c_min <= min_values_obs[c_field_name]:
                                min_values_obs[c_field_name] = c_min

                        print(F"Current max: {max_values_obs}")
                        print(F"Current min: {min_values_obs}")

    print(F"Current max: {max_values}")
    print(F"Current min: {min_values}")


def main():
    # p = Pool(NUM_PROC)
    # p.map(parallel_proc, range(NUM_PROC))
    # ---------- Sequencial -------------
    parallel_proc(1)

if __name__ == '__main__':
    main()
