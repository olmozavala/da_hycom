from datetime import datetime
from config.MainConfig_2D import get_training
from config.PreprocConfig import get_preproc_config
from AI.data_generation.GeneratorRaw2D import data_gen_from_raw

from constants_proj.AI_proj_params import ProjTrainingParams, ParallelParams, NetworkTypes, PreprocParams
from constants.AI_params import TrainingParams, ModelParams, AiModels

import trainingutils as utilsNN
# import models.modelBuilder3D as model_builder
from models.modelSelector import select_2d_model
from models_proj.models import *
from img_viz.common import create_folder
from io_project.read_utils import get_preproc_increment_files

from os.path import join
import numpy as np
import os
from constants.AI_params import TrainingParams, ModelParams

from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import LeakyReLU

def doTraining(config):
    preproc_config = get_preproc_config()
    input_folder_increment =    preproc_config[PreprocParams.input_folder_tsis]

    fields = config[ProjTrainingParams.fields_names]
    fields_obs = config[ProjTrainingParams.fields_names_obs]
    output_fields = config[ProjTrainingParams.output_fields]

    output_folder = config[TrainingParams.output_folder]
    val_perc = config[TrainingParams.validation_percentage]
    test_perc = config[TrainingParams.test_percentage]
    eval_metrics = config[TrainingParams.evaluation_metrics]
    loss_func = config[TrainingParams.loss_function]
    batch_size = config[TrainingParams.batch_size]
    epochs = config[TrainingParams.epochs]
    run_name = config[TrainingParams.config_name]
    optimizer = config[TrainingParams.optimizer]

    output_folder = join(output_folder, run_name)
    split_info_folder = join(output_folder, 'Splits')
    parameters_folder = join(output_folder, 'Parameters')
    weights_folder = join(output_folder, 'models')
    logs_folder = join(output_folder, 'logs')
    create_folder(split_info_folder)
    create_folder(parameters_folder)
    create_folder(weights_folder)
    create_folder(logs_folder)

    # Compute how many cases
    all_increment_files = os.listdir(input_folder_increment)
    files_to_read = np.array([x for x in all_increment_files if x.find(".a") != -1])
    files_to_read.sort()
    tot_examples = len(files_to_read)

    # ================ Split definition =================
    [train_ids, val_ids, test_ids] = utilsNN.split_train_validation_and_test(tot_examples,
                                                                             val_percentage=val_perc,
                                                                             test_percentage=test_perc,
                                                                             shuffle_ids=True)

    print(F"Train examples (total:{len(train_ids)}) :{files_to_read[train_ids]}")
    print(F"Validation examples (total:{len(val_ids)}) :{files_to_read[val_ids]}:")
    print(F"Test examples (total:{len(test_ids)}) :{files_to_read[test_ids]}")

    print("Selecting and generating the model....")
    now = datetime.utcnow().strftime("%Y_%m_%d_%H_%M")
    model_name = F'{run_name}_{now}'

    # ******************* Selecting the model **********************
    model = select_2d_model(config, last_activation=None)

    plot_model(model, to_file=join(output_folder,F'{model_name}.png'), show_shapes=True)

    print("Saving split information...")
    file_name_splits = join(split_info_folder, F'{model_name}.txt')
    utilsNN.save_splits(file_name=file_name_splits, train_ids=train_ids, val_ids=val_ids, test_ids=test_ids)

    print("Compiling model ...")
    model.compile(loss=loss_func, optimizer=optimizer, metrics=eval_metrics)

    print("Getting callbacks ...")

    [logger, save_callback, stop_callback] = utilsNN.get_all_callbacks(model_name=model_name,
                                                                       early_stopping_func=F'val_{eval_metrics[0].__name__}',
                                                                       weights_folder=weights_folder,
                                                                       logs_folder=logs_folder)

    print("Training ...")
    # # ----------- Using preprocessed data -------------------
    examples_per_figure = config[TrainingParams.batch_size]
    generator_train = data_gen_from_raw(config, preproc_config, train_ids, fields, fields_obs, output_fields, examples_per_figure=examples_per_figure)
    generator_val = data_gen_from_raw(config, preproc_config, val_ids, fields, fields_obs, output_fields, examples_per_figure=1)

    model.fit_generator(generator_train, steps_per_epoch=min(1000, len(train_ids)),
                        validation_data=generator_val,
                        validation_steps=min(100, len(val_ids)),
                        # validation_steps=100,
                        use_multiprocessing=False,
                        workers=1,
                        # validation_freq=10, # How often to compute the validation loss
                        epochs=epochs, callbacks=[logger, save_callback, stop_callback])

def multipleRuns(config, start_i, N, bboxes, network_types, network_names, perc_ocean):
    orig_name = config[TrainingParams.config_name]
    depth = len(config[ProjTrainingParams.fields_names]) + len(config[ProjTrainingParams.fields_names_obs]) + len(config[ProjTrainingParams.fields_names_var])
    config[ModelParams.INPUT_SIZE][2] = depth

    print(" --------------- Testing different bbox selections -------------------")
    local_name = orig_name.replace("OUTPUT", "OUT_SRFHGT")
    for i in range(N):
        for j, net_type_id in enumerate(network_types):
            for c_bbox in bboxes:
                for c_perc_ocean in perc_ocean:
                    local_name = local_name.replace("RUN", F"{(i+start_i):04d}")
                    # Set network to use
                    local_name = local_name.replace("NETWORK", network_names[j])
                    config[ProjTrainingParams.network_type] = net_type_id
                    # Set bbox to use
                    local_name = local_name.replace("ROWS", str(c_bbox[0]))
                    local_name = local_name.replace("COLS", str(c_bbox[1]))
                    input_size = config[ModelParams.INPUT_SIZE]
                    input_size[0] = c_bbox[0]
                    input_size[1] = c_bbox[1]
                    config[ModelParams.INPUT_SIZE] = input_size
                    # Set perc ocean
                    local_name = local_name.replace("PERCOCEAN", F"PERCOCEAN_{c_perc_ocean}")
                    config[ProjTrainingParams.perc_ocean] = c_perc_ocean
                    # Changing the number of images generated per example to improve speed
                    if c_bbox[0] == 80 or c_bbox[0] == 160:
                        config[TrainingParams.batch_size] = 10
                    else:
                        config[TrainingParams.batch_size] = 1

                    print(F"Final local_name: {local_name}")
                    config[TrainingParams.config_name] = local_name
                    doTraining(config)

if __name__ == '__main__':
    orig_config = get_training()
    # Single training
    # doTraining(orig_config)

    bboxes = [[160, 160]]
    perc_ocean = [.95]
    network_types = [NetworkTypes.UNET]
    network_names = ["NET_2DUNET"]
    multipleRuns(orig_config, 0, 1, bboxes, network_types, network_names, perc_ocean)

    # # ====================================================================
    # # ====================== Multiple trainings ==========================
    # # ====================================================================
    # start_i = 0 # When to start (if we already have some runs)
    # N = 5  # How many networks we want to run for each experiment
    #
    # # ========== Testing BBOX options =================
    # bboxes = [[80,80], [160,160], [384,520]]
    # perc_ocean = [0]
    # network_types = [NetworkTypes.UNET]
    # network_names = ["NET_2DUNET"]
    # multipleRuns(orig_config, start_i, N, bboxes, network_types, network_names, perc_ocean)
    #
    # # ========== Testing Types of NN options =================
    # bboxes = [[160,160]]
    # perc_ocean = [0]
    # network_types = [NetworkTypes.UNET, NetworkTypes.SimpleCNN_2, NetworkTypes.SimpleCNN_4, NetworkTypes.SimpleCNN_8, NetworkTypes.SimpleCNN_16]
    # network_names = ["NET_2DUNET", "SimpleCNN_02", "SimpleCNN_04", "SimpleCNN_08", "SimpleCNN_16"]
    # multipleRuns(orig_config, start_i, N, bboxes, network_types, network_names, perc_ocean)
    #
    # # ========== Testing perc of oceans =================
    # bboxes = [[160,160]]
    # perc_ocean = [0, .5, .9]
    # network_types = [NetworkTypes.UNET]
    # network_names = ["NET_2DUNET"]
    # multipleRuns(orig_config, start_i, N, bboxes, network_types, network_names, perc_ocean)