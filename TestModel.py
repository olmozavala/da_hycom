import os
from inout.io_netcdf import read_netcdf
from os.path import join, exists
import numpy as np

from timeseries_viz.scatter import TimeSeriesVisualizer
from config.MainConfig import get_prediction_params
from constants_proj.AI_proj_params import PredictionParams, ProjTrainingParams
from models.modelSelector import select_2d_model

from constants_proj.AI_proj_params import MAX_DA, MAX_OBS, MIN_OBS, MIN_DA

from sklearn.metrics import mean_squared_error

def main():
    config = get_prediction_params()
    test_model(config)

def test_model(config):
    input_folder = config[PredictionParams.input_folder]
    output_folder = config[PredictionParams.output_folder]
    output_field = config[ProjTrainingParams.output_field_name]
    model_weights_file = config[PredictionParams.model_weights_file]
    output_imgs_folder = config[PredictionParams.output_imgs_folder]
    day_to_predict = config[ProjTrainingParams.prediction_time]
    field_names = config[ProjTrainingParams.fields_names]
    obs_field_names = config[ProjTrainingParams.fields_names_obs]

    # Builds the visualization object
    viz_obj = TimeSeriesVisualizer(disp_images=config[PredictionParams.show_imgs],
                                     output_folder=output_imgs_folder)

    # *********** Chooses the proper model ***********
    print('Reading model ....')
    model = select_2d_model(config)

    # *********** Reads the weights***********
    print('Reading weights ....')
    model.load_weights(model_weights_file)

    # *********** Read files to predict***********
    all_files = os.listdir(input_folder)
    da_files = np.array([x for x in all_files if x.startswith('hycom')])
    obs_files = np.array([x for x in all_files if x.startswith('obs')])

    for c_file in da_files:
        # Find current and next date
        year = int(c_file.split('_')[1])
        day_of_year = int(c_file.split('_')[2].split('.')[0])

        next_day_file_name = F'hycom-tsis_{year}_{day_of_year + day_to_predict:03d}.nc'

        da_file_name = join(input_folder, c_file)
        obs_file_name = join(input_folder, [x for x in obs_files if (str(x).endswith(da_file_name[-10:]))][0])
        output_file_name = join(input_folder, next_day_file_name)

        # If the 'next day' file doesn't exist, jump to the next example
        if not (exists(output_file_name)):
            print(F"File doesn't exist: {output_file_name}")
            continue

        # *********************** Reading files **************************
        z_layers = [0]
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
            data_cube[:, :, id_field] = (input_fields_da[c_field][:rows, :cols] - MIN_DA[c_field]) / MAX_DA[c_field]
            id_field += 1

        for c_field in obs_field_names:
            data_cube[:, :, id_field] = (input_fields_obs[c_field][:rows, :cols] - MIN_OBS[c_field]) / MAX_OBS[c_field]
            id_field += 1

        # Normalizing the output "desired" date.
        y_data[:, :] = (output_field_da[output_field][:rows, :cols] - MIN_DA[output_field]) / MAX_DA[output_field]

        # Only use slices that have data (lesion inside)
        X = np.expand_dims(data_cube, axis=0)
        Y = np.expand_dims(np.expand_dims(y_data, axis=2), axis=0)

        # ******************* Replacing nan values *********
        # We set a value of 0.5 on the land. Trying a new loss function that do not takes into account land
        X = np.nan_to_num(X, nan=-0.5)
        Y = np.nan_to_num(Y, nan=-0.5)

        output_nn_all_norm = model.predict(X, verbose=1)
        land_indexes = Y == -0.5
        ocean_indexes = Y != -0.5
        output_nn_all_norm[land_indexes] = np.nan
        Y[land_indexes] = np.nan

        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 3, squeeze=True, figsize=(16 * 3, 16))
        axs[0].imshow(Y[0, :, :, 0])
        axs[0].set_title(F"Original {output_field}", fontdict={'fontsize': 80})
        img_t = axs[1].imshow(output_nn_all_norm[0, :, :, 0])
        axs[1].set_title(F"Predicted {output_field}", fontdict={'fontsize': 80})
        axs[2].imshow(output_nn_all_norm[0, :, :, 0] - Y[0, :, :, 0])
        axs[2].set_title(F"Difference MSE ~{40*mean_squared_error(output_nn_all_norm[ocean_indexes], Y[ocean_indexes]):0.4f}",
                  fontdict={'fontsize': 80})
        # fig.colorbar(img_t, ax=axs)
        # fig.suptitle(F"{c_file}", fontsize=80)
        plt.show()

        # # T
        # ax = plt.subplot(2, num_fields, num_fields + 1)
        # ax.imshow(Y[0, :, :, 0])
        # plt.show()
        #
        # plt.imshow(Y[0, :, :, 0] - X[0, :, :, 0])
        # plt.show()
        #
        # mse_mft = mean_squared_error(mft,obs)
        # mse_nn = mean_squared_error(output_nn_all_original,obs)
        # print(F'MSE MFT: {mse_mft}  NN: {mse_nn}')
        #

        # if sa:
        #     print('\t Saving Prediction...')
        #     # TODO at some point we will need to see if we can output more than one ctr
        #     data_df_original.to_csv(join(output_folder,output_file_name))

#             if compute_metrics:
#                 # Compute metrics
#                 print('\t Computing metrics....')
#                 for c_metric in metrics_params:  # Here we can add more metrics
#                     if c_metric == ClassificationMetrics.DSC_3D:
#                         metric_value = numpy_dice(output_nn_np, ctrs_np[0])
#                         data.loc[current_folder][c_metric.value] = metric_value
#                         print(F'\t\t ----- DSC: {metric_value:.3f} -----')
#                         if compute_original_resolution:
#                             metric_value = numpy_dice(output_nn_original_np,
#                                                       sitk.GetArrayViewFromImage(gt_ctr_original_itk))
#                             data.loc[current_folder][F'{ORIGINAL_TXT}_{c_metric.value}'] = metric_value
#                             print(F'\t\t ----- DSC: {metric_value:.3f} -----')
#
#                 # Saving the results every 10 steps
#                 if id_folder % 10 == 0:
#                     save_metrics_images(data, metric_names=list(metrics_dict.values()), viz_obj=viz_obj)
#                     data.to_csv(join(output_folder, output_file_name))
#
#             if save_imgs:
#                 print('\t Plotting images....')
#                 plot_intermediate_results(current_folder, data_columns, imgs_itk=imgs_itk[0],
#                                           gt_ctr_itk=ctrs_itk[0][0], nn_ctr_itk=output_nn_itk, data=data,
#                                           viz_obj=viz_obj, slices=save_imgs_slices, compute_metrics=compute_metrics)
#                 if compute_original_resolution:
#                     plot_intermediate_results(current_folder, data_columns, imgs_itk=[img_original_itk],
#                                               gt_ctr_itk=gt_ctr_original_itk,
#                                               nn_ctr_itk=output_nn_original_itk, data=data,
#                                               viz_obj=viz_obj, slices=save_imgs_slices, compute_metrics=compute_metrics,
#                                               prefix_name=ORIGINAL_TXT)
#         except Exception as e:
#             print("---------------------------- Failed {} error: {} ----------------".format(current_folder, e))
#         print(F'\t Done! Elapsed time {time.time()-t0:0.2f} seg')
#
#     if compute_metrics:
#         save_metrics_images(data, metric_names=list(metrics_dict.values()), viz_obj=viz_obj)
#         data.to_csv(join(output_folder, output_file_name))
#
#

if __name__ == '__main__':
    main()
