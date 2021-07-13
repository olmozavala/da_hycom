from info.info_hycom import read_field_names
from hycom.io import read_hycom_fields, read_field_names, read_hycom_coords
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
from config.PreprocConfig import get_preproc_config

# Not sure how to move this inside the function
NUM_PROC = 1

_minlat = 7.00250
_minlon = -98.08
_maxlat = 31.9267
_maxlon = -56.08

# Here we identify lat and lons before hand. TODO improve this making it local to functions
coords_file = "/data/HYCOM/DA_HYCOM_TSIS/preproc/coords/regional.grid.a"
# coord_fields = [ 'plon','plat','qlon','qlat','ulon','ulat','vlon','vlat']
coord_fields = ['plon','plat']
print("Reading hycom coordinates....")
hycom_coords = read_hycom_coords(coords_file, coord_fields)
_lats = hycom_coords['plat'][:,0]
_lons = hycom_coords['plon'][0,:]
print("Done!")

def preproc_data(proc_id):
    """
    This function preprocess the desired data. It does the following:
        1) Looks for dates where there is 'increment', model, and observations data.
        2) Saves the files on the same folder with only the 'desired' fields in netcdf format
    :param proc_id:
    :return:
    """
    print("Preprocessing data....")
    config = get_preproc_config()
    input_folder_increment = config[PreprocParams.input_folder_tsis]
    input_folder_model = config[PreprocParams.input_folder_hycom]
    input_folder_obs = config[PreprocParams.input_folder_obs]
    output_folder = config[PreprocParams.output_folder]
    YEARS = config[PreprocParams.YEARS]
    MONTHS = config[PreprocParams.MONTHS]
    fields = config[PreprocParams.fields_names]
    obs_fields = config[PreprocParams.fields_names_obs]
    layers = config[PreprocParams.layers_to_plot]
    img_viz = EOAImageVisualizer(output_folder=output_folder, disp_images=False)

    # These are the data assimilated files
    for c_year in YEARS:
        for c_month in MONTHS:
            print(F"=============== Year: {c_year}  Month: {c_month} ===========")
            days_of_month, days_of_year = get_days_from_month(c_month)
            # Rads all the files for this month
            da_files, da_paths = get_hycom_file_name(input_folder_increment, c_year, c_month)
            hycom_files, hycom_paths = get_hycom_file_name(input_folder_model, c_year, c_month)
            obs_files, obs_paths = get_obs_file_names(input_folder_obs, c_year, c_month)

            # This for is fixed to be able to run in parallel
            for c_day_of_month, c_day_of_year in enumerate(days_of_year):
                if (c_day_of_month % NUM_PROC) == proc_id:
                    re_increment = F'incupd.{c_year}_{c_day_of_year:03d}\S*.a'
                    re_model = F'archv.{c_year}_{c_day_of_year:03d}\S*.a'
                    re_obs = F'tsis_obs_ias_{c_year}{c_month:02d}{c_day_of_month+1:02d}\S*.nc'

                    try:
                        da_file_idx = [i for i, file in enumerate(da_files) if re.search(re_increment, file) != None][0]
                        print(F" =============== Working with: {da_files[da_file_idx]} Proc_id={proc_id} ============= ")
                        da_np_fields = read_hycom_fields(da_paths[da_file_idx], fields, layers=layers)

                        hycom_file_idx = [i for i, file in enumerate(hycom_files) if re.search(re_model, file) != None][0]
                        hycom_np_fields = read_hycom_fields(hycom_paths[hycom_file_idx], fields, layers=layers)

                        # --------- Preprocessing Increment (TSIS) -------------
                        proc_increment_data(da_np_fields, hycom_np_fields, fields,
                                            join(output_folder, F"increment_{c_year}_{c_day_of_year:03d}.nc"))
                    except Exception as e:
                        print(F"Warning: Increment file for date {c_year}-{c_month}-{c_day_of_month} ({re_increment}) doesn't exist: {e}")
                        # Only when the increment file is not found we go to the next day.
                        continue

                    try:
                        print(F" --------------- Working with: {hycom_files[hycom_file_idx]} ------------- ")
                        hycom_file_idx = [i for i, file in enumerate(hycom_files) if re.search(re_model, file) != None][0]
                        hycom_np_fields = read_hycom_fields(hycom_paths[hycom_file_idx], fields, layers=layers)
                        # --------- Preprocessing HYCOM data -------------
                        proc_model_data(hycom_np_fields, fields, join(output_folder, F"model_{c_year}_{c_day_of_year:03d}.nc"))
                    except Exception as e:
                        print(F"Warning: HYCOM file for date {c_year}-{c_month}-{c_day_of_month} ({re_model}) doesn't exist: {e}")

                    try:
                        obs_file_idx = [i for i, file in enumerate(obs_files) if re.search(re_obs, file) != None][0]
                        # --------- Preprocessing observed data -------------
                        print(F" --------------- Working with: {hycom_files[hycom_file_idx]} ------------- ")
                        obs_ds = xr.load_dataset(obs_paths[obs_file_idx])
                        for id_field, c_obs_field in enumerate(obs_fields):
                            if id_field == 0:
                                preproc_obs_ds = obs_ds[c_obs_field].to_dataset()
                            else:
                                preproc_obs_ds = preproc_obs_ds.merge(obs_ds[c_obs_field].to_dataset())

                        # --------------- Here we add the fields from the profiles as gridded data -----------
                        temp_group = 0
                        saln_group = 1
                        sst_p = np.zeros(preproc_obs_ds[c_obs_field].values.shape)
                        sss_p = np.zeros(sst_p.shape)
                        profiles = obs_ds.val
                        tot_profiles = profiles.shape[0]
                        obs_groups = obs_ds.ob_grp_present

                        lons_i = obs_ds.grdi.values[:, 0, 0]
                        lats_i = obs_ds.grdj.values[:, 0, 0]
                        for i_group, c_type in enumerate(obs_groups):
                            if c_type == saln_group or c_type == temp_group:
                                for c_profile_i in range(tot_profiles):
                                    c_data = profiles[c_profile_i, -1, i_group]
                                    if c_type == saln_group:
                                        sss_p[int(lats_i[c_profile_i]), int(lons_i[c_profile_i])] = c_data
                                    if c_type == temp_group:
                                        sst_p[int(lats_i[c_profile_i]), int(lons_i[c_profile_i])] = c_data
                        print(F"Max value: {np.amax(sst_p)}")
                        print(F"Max value s: {np.amax(sss_p)}")
                        preproc_obs_ds['sst_p'] = xr.DataArray(sst_p, dims=['yc', 'xc'])
                        preproc_obs_ds['sss_p'] = xr.DataArray(sss_p, dims=['yc', 'xc'])
                        preproc_obs_ds.to_netcdf(join(output_folder, F"obs_{c_year}_{c_day_of_year:03d}.nc"))
                    except Exception as e:
                        print(F"Warning: OBS file for date {c_year}-{c_month}-{c_day_of_month} doesn't exist: {e}")

def proc_increment_data(increment_fields, model_fields, field_names, file_name):
    """
    Preprocess the increment data. It removes the corresponding model data from the increment.
    :param increment_fields:
    :param model_fields:
    :param field_names:
    :param file_name:
    :return:
    """

    rows = increment_fields[field_names[0]].shape[1]
    cols = increment_fields[field_names[0]].shape[2]

    # lats = np.linspace(_minlat, _maxlat, rows)
    # lons = np.linspace(_minlon, _maxlon, cols)
    lats = _lats
    lons = _lons
    for id_field, c_field in enumerate(field_names):
        # df_var = {c_field: (("lat", "lon"), increment_fields[c_field][0])}
        df_var = {c_field: (("lat", "lon"), (increment_fields[c_field][0] - model_fields[c_field][0]))}

        if id_field == 0:
            preproc_increment_ds = xr.Dataset(df_var, {"lat": lats, "lon": lons})
        else:
            temp = xr.Dataset(df_var, {"lat": lats, "lon": lons})
            preproc_increment_ds = preproc_increment_ds.merge(temp)

    preproc_increment_ds = addLatLon(preproc_increment_ds , lats, lons, rows, cols)
    preproc_increment_ds.to_netcdf(file_name)

def addLatLon(ds, lats, lons, rows, cols):
    # Adding lat and lon as additional fields
    # " ---- LON -----"
    df_var = {"LON": (("lat", "lon"), lons * np.ones((rows,1)))}
    temp = xr.Dataset(df_var, {"lat": lats, "lon": lons})
    ds= ds.merge(temp)
    # " ---- LAT -----"
    df_var = {"LAT": (("lat", "lon"), (lats * np.ones((cols,1))).T)}
    temp = xr.Dataset(df_var, {"lat": lats, "lon": lons})
    ds= ds.merge(temp)
    return ds

def proc_model_data(np_fields, field_names, file_name):
    rows = np_fields[field_names[0]].shape[1]
    cols = np_fields[field_names[0]].shape[2]

    # lats = np.linspace(_minlat, _maxlat, rows)
    # lons = np.linspace(_minlon, _maxlon, cols)
    lats = _lats
    lons = _lons
    for id_field, c_field in enumerate(field_names):
        df_var = {c_field: (("lat", "lon"), np_fields[c_field][0])}

        if id_field == 0:
            preproc_model_ds = xr.Dataset(df_var, {"lat": lats, "lon": lons})
        else:
            temp = xr.Dataset(df_var, {"lat": lats, "lon": lons})
            preproc_model_ds = preproc_model_ds.merge(temp)

    preproc_model_ds = addLatLon(preproc_model_ds, lats, lons, rows, cols)
    preproc_model_ds.to_netcdf(file_name)

if __name__ == '__main__':
    # ----------- Parallel -------
    # p = Pool(NUM_PROC)
    # p.map(preproc_data, range(NUM_PROC))
    preproc_data(0)
