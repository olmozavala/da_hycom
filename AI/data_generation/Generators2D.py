import numpy as np
from datetime import date, datetime, timedelta
from inout.io_hycom import read_hycom_output
from inout.io_netcdf import read_netcdf
from os.path import join, exists
from preproc.UtilsDates import get_month_and_day_of_month_from_day_of_year, get_day_of_year_from_month_and_day

# This code is just for debugging purposes (plot intermediate steps)
from img_viz.eoa_viz import EOAImageVisualizer
from img_viz.constants import PlotMode
img_viz = EOAImageVisualizer(output_folder='/data/HYCOM/DA_HYCOM_TSIS/images/inputNN', disp_images=False)

MAX_DA = {'temp': 40, 'srfhgt': 20, 'salin': 70, 'u-vel.': 4, 'v-vel.': 4}
MIN_DA = {'temp': 0, 'srfhgt': -20, 'salin': 0, 'u-vel.': -4, 'v-vel.': -4}

MAX_OBS = {'sst': 40, 'ssh': 0.9, 'sss': 40}
MIN_OBS = {'sst': 0,  'ssh':-0.9, 'sss': 15}

def data_gen_hycomtsis(paths, file_names, obs_path,
                       field_names, obs_field_names, output_field, days_separation=1, z_layers=[0]):
    """
    This generator should generate X and Y for a CNN
    :param path:
    :param file_names:
    :return:
    """
    ex_id = -1
    ids = np.arange(len(file_names))
    while True:
        # These lines are for sequential selection
        if ex_id < (len(ids) - 1):
            ex_id += 1
        else:
            ex_id = 0
            np.random.shuffle(ids) # We shuffle the folders every time we have tested all the examples

        file_name = join(paths[ex_id], file_names[ex_id])
        date_str = file_names[ex_id].split('.')[1]  # This should be the date
        date_array = date_str.split('_')

        year = int(date_array[0])
        day_of_year = int(date_array[1])
        month, day_month = get_month_and_day_of_month_from_day_of_year(day_of_year, year)

        # Verify next time exist
        cur_date = date(year, month, day_month)
        desired_date = date(year, month, day_month) + timedelta(days=days_separation)
        desired_file_name = F'archv.{desired_date.year}_{get_day_of_year_from_month_and_day(desired_date.month, desired_date.day):03d}_00.a'

        if not(exists(join(paths[ex_id],desired_file_name))):
            print(F"Warning! File {desired_file_name} doesn't exist")
            continue

        # try:
        # *********************** Reading DA files **************************
        input_fields_da = read_hycom_output(file_name, field_names, layers=z_layers)
        output_field_da = read_hycom_output(join(paths[ex_id],desired_file_name), [output_field], layers=z_layers)

        # *********************** Reading Obs file **************************
        # TODO Hardcoded text "WITH_PIES"
        obs_file_name = join(obs_path, "WITH_PIES", F"tsis_obs_ias_{desired_date.year}{desired_date.month:02d}{desired_date.day:02d}00.nc")
        if not (exists(obs_file_name)):
            print(F"Warning! Observation file doesn't exist {obs_file_name}")
            continue

        # ******************* Normalizing and Cropping Data *******************
        # TODO hardcoded dimensions and cropping code
        input_fields_obs = read_netcdf(obs_file_name, obs_field_names, z_layers)
        # dims = input_fields_da[field_names[0]].shape
        rows = 888
        cols = 1400
        num_fields = 8

        data_cube = np.zeros((rows, cols, num_fields))

        id_field = 0
        for c_field in field_names:
            # data_cube[id_field, :, :] = input_fields_da[c_field][0, :, :]
            data_cube[:, :, id_field] = (input_fields_da[c_field][0, :rows, :cols] - MIN_DA[c_field])/MAX_DA[c_field]
            id_field += 1

        for c_field in obs_field_names:
            # if len(input_fields_obs[c_field].shape) == 3:
                # data_cube[id_field, :, :] = input_fields_obs[c_field][0, :, :]
                # data_cube[:, :, id_field] = input_fields_obs[c_field][0, :rows, :cols]
            if len(input_fields_obs[c_field].shape) == 2:
                # data_cube[id_field, :, :] = input_fields_obs[c_field][:, :]
                data_cube[:, :, id_field] = (input_fields_obs[c_field][:rows, :cols] - MIN_OBS[c_field])/MAX_OBS[c_field]
            id_field += 1

        # ******************* Replacing nan values *********

        # Only use slices that have data (lesion inside)
        X = np.expand_dims(data_cube, axis=0)
        Y = np.expand_dims(np.expand_dims(output_field_da[output_field][0, :rows, :cols],axis=2), axis=0)

        X = np.nan_to_num(X, nan=-1)
        Y = np.nan_to_num(Y, nan=-1)

        # img_viz.plot_3d_data_singlevar_np(np.swapaxes(np.swapaxes(X[0],0,2), 1,2),
        #                               z_levels=range(len(field_names+obs_field_names)),
        #                                title='Input NN',
        #                                file_name_prefix=F'{year}_{month:02d}_{day_month:02d}',
        #                                   flip_data=True)
        #
        # img_viz.plot_3d_data_singlevar_np(np.swapaxes(np.swapaxes(Y[0],0,2), 1,2),
        #                               z_levels=[0],
        #                                title='Input NN',
        #                                file_name_prefix=F'output_{year}_{month:02d}_{day_month:02d}',
        #                                   flip_data=True)

        yield X, Y
        # except Exception as e:
        #     print("----- Not able to generate for: ", 1, " ERROR: ", str(e))
