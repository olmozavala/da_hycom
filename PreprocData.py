from info.info_hycom import read_field_names
from inout.io_hycom import read_hycom_output
from inout.io_netcdf import read_netcdf
from img_viz.eoa_viz import EOAImageVisualizer
from img_viz.constants import PlotMode
from preproc.UtilsDates import get_days_from_month
from preproc.metrics import mse
from io_project.read_utils import *
import xarray as xr
import re
import numpy as np
from multiprocessing import Pool
from constants_proj.AI_proj_params import PreprocParams, ParallelParams
from config.PreprocConfig import get_preproc_config, get_paralallel_config

# Not sure how to move this inside the function
config_par = get_paralallel_config()
NUM_PROC = config_par[ParallelParams.NUM_PROC]

def parallel_proc(proc_id):
    # /data/COAPS_nexsan/people/abozec/TSIS/IASx0.03/obs/qcobs_mdt_gofs/WITH_PIES
    config = get_preproc_config()
    input_folder_tsis = config[PreprocParams.input_folder_tsis]
    input_folder_obs = config[PreprocParams.input_folder_obs]
    output_folder = config[PreprocParams.output_folder]
    PIES = config[PreprocParams.PIES]
    YEARS = config[PreprocParams.YEARS]
    MONTHS = config[PreprocParams.MONTHS]
    fields = config[PreprocParams.fields_names]
    obs_fields = config[PreprocParams.fields_names_obs]
    layers = config[PreprocParams.layers_to_plot]
    img_viz = EOAImageVisualizer(output_folder=output_folder, disp_images=False)

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
                            obs_file_idx = [i for i, file in enumerate(obs_files)
                                            if re.search(re_obs, file) != None][0]

                        except Exception as e:
                            print(F"ERROR: The file for date {c_year} - {c_month} - {c_day_of_month} doesn't exist: {e}")
                            continue

                        print(F" =============== Working with: {da_files[da_file_idx]} ============= ")
                        da_np_fields = read_hycom_output(da_paths[da_file_idx], fields, layers=layers)

                        # --------- Preprocessing HYCOM-TSIS data -------------
                        for id_field, c_field in enumerate(fields):
                            rows = da_np_fields[c_field].shape[1]
                            cols = da_np_fields[c_field].shape[2]
                            df_var = {c_field: (("lat", "lon"), da_np_fields[c_field][0])}

                            if id_field == 0:
                                preproc_da_ds = xr.Dataset(df_var, {"lat": np.arange(rows), "lon": np.arange(cols)})
                            else:
                                temp = xr.Dataset(df_var, {"lat": np.arange(rows), "lon": np.arange(cols)})
                                preproc_da_ds = preproc_da_ds.merge(temp)
                        preproc_da_ds.to_netcdf(join(output_folder, F"hycom-tsis_{c_year}_{c_day_of_year:03d}.nc"))

                        # --------- Preprocessing observed data -------------
                        obs_ds = xr.load_dataset(obs_paths[obs_file_idx])
                        for id_field, c_obs_field in enumerate(obs_fields):
                            if id_field == 0:
                                preproc_obs_ds = obs_ds[c_obs_field].to_dataset()
                            else:
                                preproc_obs_ds = preproc_obs_ds.merge(obs_ds[c_obs_field].to_dataset())
                        preproc_obs_ds.to_netcdf(join(output_folder, F"obs_{c_year}_{c_day_of_year:03d}.nc"))


def main():
    # ----------- Parallel -------
    p = Pool(NUM_PROC)
    p.map(parallel_proc, range(NUM_PROC))

    # ----------- Sequencial -------
    # NUM_PROC = 1
    # parallel_proc(1)

if __name__ == '__main__':
    main()
