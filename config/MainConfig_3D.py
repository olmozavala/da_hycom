from tensorflow.keras.optimizers import Adam, SGD
import tensorflow.keras.metrics as metrics
import tensorflow.keras.losses as losses
from os.path import join
import os
import tensorflow as tf

import AI.proj_metrics as mymetrics
from constants_proj.AI_proj_params import *
from constants.AI_params import TrainingParams, ModelParams, AiModels
from img_viz.constants import PlotMode

# ----------------------------- UM -----------------------------------
# _run_name = F'UNET_Input_All_wvar_NO_LATLON_Output_only_VAR_160x160_UpSampling_NoLand'
# _run_name = F'RUN_STDNORM_NETWORK_IN_No-STD_OUTPUT_160x160'
# _run_name = F'GoM3D_NONORM_NETWORK_IN_No-STD_OUTPUT_ROWSxCOLSxDEPTH'  # No-STD means not adding the VARIANCE from the model TSIS remember?
_run_name = F'GoM3D_NONORM_3DUNET_IN7_plus_obsSST_No-STD_OUTPUT_SSH_ROWSxCOLSxDEPTH'  # No-STD means not adding the VARIANCE from the model TSIS remember?
# _run_name = F'RUN_NETWORK_IN_OUTPUT_160x160'
# _run_name = F'DELETERUN_NETWORK_IN_No-STD_OUTPUT_160x160'

_output_folder = '/data/HYCOM/DA_HYCOM_TSIS/'  # Where to save everything
_preproc_folder = "/data/HYCOM/DA_HYCOM_TSIS/preproc"

# _preproc_folder = '/home/data/MURI/preproc'
# _output_folder = '/home/data/MURI/output'

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Decide which GPU to use to execute the code
# tf.config.experimental.VirtualDeviceConfiguration(memory_limit=12288)
tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2000)

train_rows = 384  # 385, 525
train_cols = 520 # 385, 525

def append_model_params(cur_config):
    bands = len(cur_config[ProjTrainingParams.fields_names]) + len(cur_config[ProjTrainingParams.fields_names_obs])
    depth = 24
    output_fields = len(cur_config[ProjTrainingParams.output_fields])
    model_config = {
        ModelParams.MODEL: AiModels.UNET_3D_SINGLE,
        ModelParams.DROPOUT: False,
        ModelParams.BATCH_NORMALIZATION: True,
        ModelParams.INPUT_SIZE: [train_rows, train_cols, depth, bands],  # 4741632
        ModelParams.START_NUM_FILTERS: 16,
        ModelParams.NUMBER_LEVELS: 2,
        ModelParams.FILTER_SIZE: 3,
        ModelParams.OUTPUT_SIZE: output_fields
    }
    return {**cur_config, **model_config}


def appendFields(cur_config):
    model_config = {
        # ProjTrainingParams.fields_names:   ['u-vel.', 'v-vel.','temp', 'salin', 'thknss', 'srfhgt', 'mix_dpth'],
        ProjTrainingParams.fields_names:   ['u-vel.', 'v-vel.','temp', 'salin', 'thknss', 'srfhgt', 'mix_dpth'],
        ProjTrainingParams.fields_names_obs: ['sst'],
        # ProjTrainingParams.fields_names_var: ['tem', 'sal', 'ssh', 'mdt'],
        ProjTrainingParams.fields_names_var: [],
        # ProjTrainingParams.output_fields: ['u-vel.', 'v-vel.','temp', 'salin', 'thknss', 'srfhgt']
        ProjTrainingParams.output_fields: ['temp', 'salin']
    }
    return {**cur_config, **model_config}


def get_training_2d():
    cur_config = {
        TrainingParams.output_folder: F"{join(_output_folder,'Training')}",
        TrainingParams.validation_percentage: .1,
        TrainingParams.test_percentage: .1,
        TrainingParams.file_name: 'RESULTS.csv',

        # TrainingParams.evaluation_metrics: [mymetrics.only_ocean_mse, mymetrics.mse],  # Metrics to show in tensor flow in the training
        # TrainingParams.loss_function: mymetrics.only_ocean_mse,  # Loss function to use for the learning
        TrainingParams.evaluation_metrics: [losses.mse],  # Metrics to show in tensor flow in the training
        TrainingParams.loss_function: losses.mse,  # Loss function to use for the learning

        TrainingParams.optimizer: Adam(lr=0.001),  # Default values lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None,
        TrainingParams.batch_size: 1, # READ In this case it is not a common batch size. It indicates the number of images to read from the same file
        TrainingParams.epochs: 5000,
        TrainingParams.config_name: _run_name,
        TrainingParams.data_augmentation: False,
        ProjTrainingParams.input_folder_preproc: '/data/HYCOM/DA_HYCOM_TSIS/preproc',
        ProjTrainingParams.output_folder: join(_output_folder, 'images'),
        ProjTrainingParams.YEARS: [2009],
        ProjTrainingParams.MONTHS: range(1, 13),
        ProjTrainingParams.rows: train_rows,
        ProjTrainingParams.cols: train_cols,
        # ProjTrainingParams.norm_type: PreprocParams.zero_one,
        ProjTrainingParams.norm_type: PreprocParams.mean_var,
        # ProjTrainingParams.prediction_time: 1,
        ProjTrainingParams.network_type: NetworkTypes.UNET,
        ProjTrainingParams.output_folder_summary_models:  F"{join(_output_folder,'SUMMARY')}",
    }
    return append_model_params(appendFields(cur_config))


def get_prediction_params():
    weights_folder = join(_output_folder,'Training', _run_name, 'models')
    cur_config = {
        TrainingParams.config_name: _run_name,
        PredictionParams.input_folder: _preproc_folder,
        PredictionParams.output_folder: F"{join(_output_folder,'Prediction')}",
        PredictionParams.output_imgs_folder: F"{join(_output_folder,'Prediction','imgs')}",
        PredictionParams.show_imgs: False,
        PredictionParams.model_weights_file: join(weights_folder, "Simple_CNNVeryLarge_Input_All_with_Obs_No_SSH_NO_LATLON_Output_ALL_80x80_UpSampling_NoLand_Mean_Var_2020_10_29_17_10-01-0.45879732.hdf5"),
        PredictionParams.metrics: mymetrics.only_ocean_mse,
    }

    return {**append_model_params(appendFields(cur_config)), **get_training_2d()}

