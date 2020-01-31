import numpy as np
from inout.io_netcdf import read_netcdf
from os.path import join, exists
from constants_proj.AI_proj_params import MAX_DA, MAX_OBS, MIN_OBS, MIN_DA


def data_gen_from_preproc(path, obs_files, da_files, field_names, obs_field_names, output_field,
                          days_separation=1, z_layers=[0]):
    """
    This generator should generate X and Y for a CNN
    :param path:
    :param file_names:
    :return:
    """
    ex_id = -1
    ids = np.arange(len(da_files))
    np.random.shuffle(ids)

    while True:
        # These lines are for sequential selection
        if ex_id < (len(ids) - 1):
            ex_id += 1
        else:
            ex_id = 0
            np.random.shuffle(ids) # We shuffle the folders every time we have tested all the examples

        try:
            # Find current and next date
            year = int(da_files[ex_id].split('_')[1])
            day_of_year = int(da_files[ex_id].split('_')[2].split('.')[0])

            next_day_file_name = F'hycom-tsis_{year}_{day_of_year+days_separation:03d}.nc'

            da_file_name = join(path, da_files[ex_id])
            obs_file_name = join(path, [x for x in obs_files if(str(x).endswith(da_file_name[-10:]))][0])
            output_file_name = join(path, next_day_file_name)

            # If the 'next day' file doesn't exist, jump to the next example
            if not(exists(output_file_name)):
                print(F"File doesn't exist: {output_file_name}")
                continue

            # *********************** Reading files **************************
            input_fields_da = read_netcdf(da_file_name, field_names, z_layers)
            input_fields_obs = read_netcdf(obs_file_name, obs_field_names, z_layers)
            output_field_da = read_netcdf(output_file_name, [output_field], z_layers)

            # ******************* Normalizing and Cropping Data *******************
            # TODO hardcoded dimensions and cropping code
            # dims = input_fields_da[field_names[0]].shape
            rows = 888
            cols = 1400
            num_fields = len(field_names) + len(obs_field_names)

            data_cube = np.zeros((rows, cols, num_fields))
            y_data = np.zeros((rows, cols))

            id_field = 0
            for c_field in field_names:
                data_cube[:, :, id_field] = (input_fields_da[c_field][:rows, :cols] - MIN_DA[c_field])/MAX_DA[c_field]
                id_field += 1

            for c_field in obs_field_names:
                data_cube[:, :, id_field] = (input_fields_obs[c_field][:rows, :cols] - MIN_OBS[c_field])/MAX_OBS[c_field]
                id_field += 1


            # ******************* Replacing nan values *********
            y_data[:,:] = (output_field_da[output_field][:rows, :cols] - MIN_DA[output_field])/MAX_DA[output_field]

            # Only use slices that have data (lesion inside)
            X = np.expand_dims(data_cube, axis=0)
            Y = np.expand_dims(np.expand_dims(y_data, axis=2), axis=0)

            # We set a value of 0.5 on the land. Trying a new loss function that do not takes into account land
            X = np.nan_to_num(X, nan=-0.5)
            Y = np.nan_to_num(Y, nan=-0.5)

            # --------------- Just for debugging ---------------------------
            # import matplotlib.pyplot as plt
            # plt.subplots(2, num_fields, squeeze=True, figsize=(16*num_fields, 16))
            # for c_field in range(num_fields):
            #     ax = plt.subplot(2, num_fields, c_field+1)
            #     ax.imshow(X[0,:,:,c_field])
            #
            # # T
            # ax = plt.subplot(2, num_fields, num_fields+1)
            # ax.imshow(Y[0, :, :, 0])
            # plt.show()
            #
            # plt.imshow(Y[0, :, :, 0]-X[0,:,:,0])
            # plt.show()

            yield X, Y
        except Exception as e:
            print("----- Not able to generate for: ", 1, " ERROR: ", str(e))
