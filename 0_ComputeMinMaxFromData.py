from info.info_hycom import read_field_names
from config.PreprocConfig import get_preproc_config
from datetime import datetime
from inout.io_hycom import read_hycom_fields
from inout.io_netcdf import read_netcdf_xr
from img_viz.eoa_viz import EOAImageVisualizer
from img_viz.constants import PlotMode
from preproc.UtilsDates import get_days_from_month
from preproc.metrics import mse
from io_project.read_utils import *
import pandas as pd
import re
import numpy as np
from multiprocessing import Pool
from constants_proj.AI_proj_params import PreprocParams, ParallelParams, ProjTrainingParams
from config.MainConfig_2D import get_training
import matplotlib.pyplot as plt

# Not sure how to move this inside the function
NUM_PROC = 10

# /data/COAPS_nexsan/people/abozec/TSIS/IASx0.03/obs/qcobs_mdt_gofs/WITH_PIES

def ComputeOverallMinMaxVar():
    """
    Computes the mean, max and variance for all the fields in the files
    :return:
    """
    config = get_training()
    input_folder = config[ProjTrainingParams.input_folder_preproc]
    fields = config[ProjTrainingParams.fields_names]
    fields_obs = config[ProjTrainingParams.fields_names_obs]

    max_values_model = {field: 0 for field in fields}
    min_values_model = {field: 10**5 for field in fields}
    max_values_obs = {field: 0 for field in fields_obs}
    min_values_obs = {field: 10**5 for field in fields_obs}
    max_values_inc = {field: 0 for field in fields}
    min_values_inc = {field: 10**5 for field in fields}

    mean_values_model = {field: 0 for field in fields}
    mean_values_obs = {field: 0 for field in fields_obs}
    mean_values_inc = {field: 0 for field in fields}

    var_values_model = {field: 0 for field in fields}
    var_values_obs = {field: 0 for field in fields_obs}
    var_values_inc = {field: 0 for field in fields}

    # These are the data assimilated files
    all_files = os.listdir(input_folder)
    all_files.sort()
    model_files = np.array([x for x in all_files if x.startswith('model')])
    model_files.sort()

    # model_files = model_files[55:58]
    tot_files = len(model_files)

    # Iterate over all the model files
    for id_file, c_file in enumerate(model_files):
        print(F"Working with {c_file}")
        # Find current and next date
        year = int(c_file.split('_')[1])
        day_of_year = int(c_file.split('_')[2].split('.')[0])

        model_file = join(input_folder, F'model_{year}_{day_of_year:03d}.nc')
        inc_file = join(input_folder,F'increment_{year}_{day_of_year:03d}.nc')
        obs_file = join(input_folder,F'obs_{year}_{day_of_year:03d}.nc')

        # *********************** Reading files **************************
        z_layers = [0]
        input_fields_model = read_netcdf(model_file, fields, z_layers)
        input_fields_obs = read_netcdf(obs_file, fields_obs, z_layers)
        output_field_increment = read_netcdf(inc_file, fields, z_layers)


        # =============== Computing max values for the model
        for idx_field, c_field_name in enumerate(fields):
            da_np_c_field = input_fields_model[c_field_name]
            # Computing mean also
            mean_values_model[c_field_name] += np.nanmean(da_np_c_field) / tot_files
            c_max = np.nanmax(da_np_c_field)
            c_min = np.nanmin(da_np_c_field)
            if c_max >= max_values_model[c_field_name]:
                max_values_model[c_field_name] = c_max
            if c_min <= min_values_model[c_field_name]:
                min_values_model[c_field_name] = c_min
        # print(F"Cur max for model: {max_values_model}")
        # print(F"Cur max for model: {min_values_model}")

        # =============== Computing max values for the observations
        for idx_field, c_field_name in enumerate(fields_obs):
            da_np_c_field = input_fields_obs[c_field_name]
            # We needed to add this try because in some cases there are none observations, like in day 245
            try:
                mean_values_obs[c_field_name] += np.nanmean(da_np_c_field) / tot_files
            except Exception as e:
                mean_values_obs[c_field_name] += 0
            print(F' {c_file}:{c_field_name}: {mean_values_obs[c_field_name]}')

            c_max = np.nanmax(da_np_c_field)
            c_min = np.nanmin(da_np_c_field)
            if c_max >= max_values_obs[c_field_name]:
                max_values_obs[c_field_name] = c_max
            if c_min <= min_values_obs[c_field_name]:
                min_values_obs[c_field_name] = c_min
        # print(F"Cur max for obs: {max_values_obs}")
        # print(F"Cur min for obs: {min_values_obs}")

        # =============== Computing max values for the increment
        for idx_field, c_field_name in enumerate(fields):
            da_np_c_field = output_field_increment[c_field_name]
            # Computing mean also
            mean_values_inc[c_field_name] += np.nanmean(da_np_c_field) / tot_files
            c_max = np.nanmax(da_np_c_field)
            c_min = np.nanmin(da_np_c_field)
            if c_max >= max_values_inc[c_field_name]:
                max_values_inc[c_field_name] = c_max
            if c_min <= min_values_inc[c_field_name]:
                min_values_inc[c_field_name] = c_min
        # print(F"Cur max for inc: {max_values_inc}")
        # print(F"Cur min for inc: {min_values_inc}")

    # Computing STD
    print("=============================== Computing Variance....")
    for id_file, c_file in enumerate(model_files):
        print(F"Working with {c_file}")
        # Find current and next date
        year = int(c_file.split('_')[1])
        day_of_year = int(c_file.split('_')[2].split('.')[0])

        model_file = join(input_folder, F'model_{year}_{day_of_year:03d}.nc')
        inc_file = join(input_folder,F'increment_{year}_{day_of_year:03d}.nc')
        obs_file = join(input_folder,F'obs_{year}_{day_of_year:03d}.nc')

        # *********************** Reading files **************************
        z_layers = [0]
        input_fields_model = read_netcdf(model_file, fields, z_layers)
        input_fields_obs = read_netcdf(obs_file, fields_obs, z_layers)
        output_field_increment = read_netcdf(inc_file, fields, z_layers)

        # =============== Computing max values for the model
        for idx_field, c_field_name in enumerate(fields):
            da_np_c_field = input_fields_model[c_field_name]
            var_values_model[c_field_name] += np.nanmean( (da_np_c_field - mean_values_model[c_field_name])**2 ) / tot_files

        # =============== Computing max values for the observations
        for idx_field, c_field_name in enumerate(fields_obs):
            da_np_c_field = input_fields_obs[c_field_name]
            data = (da_np_c_field[:].filled(np.nan) - mean_values_obs[c_field_name])**2
            if (np.logical_not(np.isnan(data)).any()):
                var_values_obs[c_field_name] += np.nanmean(data) / tot_files
            # print(F' {c_file}:{c_field_name}: {var_values_obs[c_field_name]}')

        # =============== Computing max values for the increment
        for idx_field, c_field_name in enumerate(fields):
            da_np_c_field = output_field_increment[c_field_name]
            var_values_inc[c_field_name] += np.nanmean( (da_np_c_field - mean_values_inc[c_field_name])**2 ) / tot_files

    print("----------------- Model --------------------")
    f = open("MIN_MAX_MEAN_STD.csv", 'w')
    text = F"TYPE,Field,MIN,MAX,MEAN,VARIANCE,STD\n"
    f.write(text)

    for c_field_name in fields:
        text = F"MODEL,{c_field_name},  {min_values_model[c_field_name]:0.6f},  {max_values_model[c_field_name]:0.6f}, " \
               F" {mean_values_model[c_field_name]:0.6f}, {var_values_model[c_field_name]: 0.6f}, {np.sqrt(var_values_model[c_field_name]): 0.6f}\n"
        f.write(text)
        print(text)

    print("----------------- Observations --------------------")
    for c_field_name in fields_obs:
        text = F"OBS,{c_field_name},  {min_values_obs[c_field_name]:0.6f},  {max_values_obs[c_field_name]:0.6f}, " \
            F" {mean_values_obs[c_field_name]:0.6f}, {var_values_obs[c_field_name]: 0.6f}, {np.sqrt(var_values_obs[c_field_name]): 0.6f}\n"
        f.write(text)
        print(text)

    print("----------------- Increment --------------------")
    for c_field_name in fields:
        text = F"INC,{c_field_name},  {min_values_inc[c_field_name]:0.6f},  {max_values_inc[c_field_name]:0.6f}," \
        F" {mean_values_inc[c_field_name]:0.6f}, {var_values_inc[c_field_name]: 0.6f}, {np.sqrt(var_values_inc[c_field_name]): 0.6f}\n"
        f.write(text)
        print(text)

    f.close()

def ComputeMinMaxSTDFields(file_name, fields_names, output_file):

    data = read_netcdf(file_name, [], [0])
    out_fields = []
    out_mins = []
    out_maxs = []
    out_vars = []
    out_means = []

    for field_name in fields_names:
        if len(data[field_name].shape) == 2:
            field = data[field_name][:]
        elif len(data[field_name].shape) == 3:
            field = data[field_name][0, :]

        # im = plt.imshow(np.flip(field, axis=0), cmap='gist_earth')
        # plt.colorbar(im)
        # plt.title(field_name)
        # plt.show()

        out_fields.append(field_name)
        out_mins.append(np.amin(field))
        out_maxs.append(np.amax(field))
        out_means.append(np.mean(field))
        out_vars.append(np.var(field))

    out_dic = {"Name": ["STD" for x in range(len(out_fields))],
               "Field": out_fields,
               "MIN": out_mins,
               "MAX": out_maxs,
               "MEAN": out_means,
               "VAR": out_vars,
               }

    df = pd.DataFrame.from_dict(out_dic)

    df.to_csv(output_file)

def compute_stats_from_raw():
    """
    We compute the mean and std for each field in the preproc config
    :return:
    """
    preproc_config = get_preproc_config()
    input_folder_background =   preproc_config[PreprocParams.input_folder_hycom]
    input_folder_increment =    preproc_config[PreprocParams.input_folder_tsis]
    input_folder_observations = preproc_config[PreprocParams.input_folder_obs]
    output_stats_file = preproc_config[PreprocParams.output_stats_file]

    background_fields = preproc_config[PreprocParams.fields_names]
    increment_fields = background_fields
    obs_fields = preproc_config[PreprocParams.fields_names_obs]

    background_stats = {x:{'mean':0, 'std':0} for x in background_fields}
    obs_stats = {x:{'mean':0, 'std':0} for x in obs_fields}

    # --------- Reading all file names --------------
    increment_files = np.array([join(input_folder_increment, x).replace(".a", "") for x in os.listdir(input_folder_increment) if x.endswith('.a')])
    increment_files.sort()

    # --------- Preallocating ndarrays all file names --------------
    z_layers = [0]
    temp = read_hycom_fields(increment_files[0], increment_fields, z_layers) # Only in the surfacek
    dims = temp[increment_fields[0]].shape
    increment_data = {x:np.zeros([len(increment_files), dims[1], dims[2]]) for x in increment_fields}
    obs_data = {x:np.zeros([len(increment_files),  dims[1], dims[2]]) for x in obs_fields}

    # Filling arrays with data
    for f_idx, c_file in enumerate(increment_files):
        print(F"Working with file: {c_file}")
        # Finding corresponding observation file
        sp_name = c_file.split("/")[-1].split(".")[1]
        c_datetime = datetime.strptime(sp_name, "%Y_%j_18")
        obs_file  = join(input_folder_observations, F"tsis_obs_gomb4_{c_datetime.strftime('%Y%m%d')}00.nc")
        assert os.path.exists(obs_file)

        temp = read_hycom_fields(c_file, increment_fields, z_layers) # Only in the surfacek
        if "thknss" in increment_fields:
            divide = 9806
            temp["thknss"] = temp["thknss"]/divide
        for c_field in increment_fields:
            increment_data[c_field][f_idx, :, :] = temp[c_field]

        temp = read_netcdf_xr(obs_file, obs_fields, z_layers)
        for c_field in obs_fields:
            obs_data[c_field][f_idx, :, :] = temp[c_field][:].astype(np.float64)

    # Computing and storing the statistics
    for c_field in increment_fields:
        background_stats[c_field]['mean'] = np.nanmean(increment_data[c_field][f_idx, :, :])
        background_stats[c_field]['std'] = np.nanstd(increment_data[c_field][f_idx, :, :])

    for c_field in obs_fields:
        obs_stats[c_field]['mean'] = np.nanmean(obs_data[c_field][f_idx, :, :])
        obs_stats[c_field]['std'] = np.nanstd(obs_data[c_field][f_idx, :, :])

    # Saving the statistics
    df = pd.DataFrame(background_stats)
    df.to_csv(F"{output_stats_file}_background.csv")
    df = pd.DataFrame(obs_stats)
    df.to_csv(F"{output_stats_file}_obs.csv")
    print(F"Saved: {output_stats_file}_background.csv")
    print(F"Saved: {output_stats_file}_obs.csv")
    print("Done! yeah finally!")

def main():
    # p = Pool(NUM_PROC)
    # p.map(parallel_proc, range(NUM_PROC))
    # ---------- -------------
    # ComputeOverallMinMaxVar()
    compute_stats_from_raw()
    # std_file = "/data/HYCOM/DA_HYCOM_TSIS/preproc/cov_mat/tops_ias_std.nc"
    # ComputeMinMaxSTDFields(std_file, ['tem', 'sal', 'ssh', 'mdt'], "STD_vars_min_max.csv")

if __name__ == '__main__':
    main()

