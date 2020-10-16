from datetime import datetime

from config.MainConfig import get_training_2d
# from AI.data_generation.Generators2D import data_gen_hycomtsis
from AI.data_generation.GeneratorPreproc import data_gen_from_preproc

from constants_proj.AI_proj_params import ProjTrainingParams, ParallelParams
import trainingutils as utilsNN
# import models.modelBuilder3D as model_builder
from models.modelSelector import select_2d_model
from models_proj.models import simpleCNN
from img_viz.common import create_folder
from io_project.read_utils import get_preproc_increment_files
from os.path import join
import numpy as np
import os
from constants.AI_params import TrainingParams, ModelParams

from tensorflow.keras.utils import plot_model

if __name__ == '__main__':

    config = get_training_2d()

    input_folder_preproc = config[ProjTrainingParams.input_folder_preproc]
    # input_folder_obs = config[ProjTrainingParams.input_folder_obs]
    years = config[ProjTrainingParams.YEARS]
    fields = config[ProjTrainingParams.fields_names]
    fields_obs = config[ProjTrainingParams.fields_names_obs]
    output_field = config[ProjTrainingParams.output_fields]
    # day_to_predict = config[ProjTrainingParams.prediction_time]

    output_folder = config[TrainingParams.output_folder]
    val_perc = config[TrainingParams.validation_percentage]
    test_perc = config[TrainingParams.test_percentage]
    eval_metrics = config[TrainingParams.evaluation_metrics]
    loss_func = config[TrainingParams.loss_function]
    batch_size = config[TrainingParams.batch_size]
    epochs = config[TrainingParams.epochs]
    run_name = config[TrainingParams.config_name]
    optimizer = config[TrainingParams.optimizer]

    nn_input_size = config[ModelParams.INPUT_SIZE]
    model_type = config[ModelParams.MODEL]

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
    files_to_read, paths_to_read = get_preproc_increment_files(input_folder_preproc)
    tot_examples = len(files_to_read)

    # ================ Split definition =================
    [train_ids, val_ids, test_ids] = utilsNN.split_train_validation_and_test(tot_examples,
                                                                             val_percentage=val_perc,
                                                                             test_percentage=test_perc)

    print(F"Train examples (total:{len(train_ids)}) :{files_to_read[train_ids]}")
    print(F"Validation examples (total:{len(val_ids)}) :{files_to_read[val_ids]}:")
    print(F"Test examples (total:{len(test_ids)}) :{files_to_read[test_ids]}")

    print("Selecting and generating the model....")
    now = datetime.utcnow().strftime("%Y_%m_%d_%H_%M")
    model_name = F'{run_name}_{now}'

    # ******************* Selecting the model **********************
    model = select_2d_model(config, last_activation='relu')
    # model = simpleCNN(config)

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
    # ----------- Using preprocessed data -------------------
    generator_train = data_gen_from_preproc(input_folder_preproc, config, train_ids, fields, fields_obs, output_field)
    generator_val = data_gen_from_preproc(input_folder_preproc, config, val_ids, fields, fields_obs, output_field)

    # Decide which generator to use
    data_augmentation = config[TrainingParams.data_augmentation]

    model.fit_generator(generator_train, steps_per_epoch=min(100, len(train_ids)),
                        validation_data=generator_val,
                        validation_steps=min(20, len(val_ids)),
                        use_multiprocessing=False,
                        workers=1,
                        # validation_freq=10, # How often to compute the validation loss
                        epochs=epochs, callbacks=[logger, save_callback, stop_callback])
