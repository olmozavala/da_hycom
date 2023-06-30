from tensorflow.keras.optimizers import Adam, SGD
import tensorflow.keras.metrics as metrics
import tensorflow.keras.losses as losses
from os.path import join
import os
import tensorflow as tf

import AI.proj_metrics as mymetrics
from constants_proj.AI_proj_params import *
from ai_common.constants.AI_params import TrainingParams, ModelParams, AiModels

# _run_name = F'GoM2D_STDNORM_PERCOCEAN_NETWORK_IN8_No-STD_OUTPUT_ROWSxCOLS'  # No-STD means not adding the VARIANCE from the model TSIS remember?
_run_name = F'RUN_GoM2D_STDNORM_PERCOCEAN_NETWORK_ININ_OBSIN_No-STD_OUTPUT_ROWSxCOLS'  # No-STD means not adding the VARIANCE from the model TSIS remember?
# _run_name = F'0002_GoM2D_STDNORM_PERCOCEAN_0_NET_2DUNET_srfhgt_ssh-ssh-err-sst-sst-err_No-STD_OUT_SRFHGT_384x520'  # Best run 2022

_output_folder = '/data/HYCOM/DA_HYCOM_TSIS/'  # Where to save everything
# _output_folder = '/home/olmozavala/DAN_HYCOM/OUTPUT/'
_preproc_folder = "/data/HYCOM/DA_HYCOM_TSIS/preproc"

# _preproc_folder = '/home/data/MURI/preproc'
# _output_folder = '/home/data/MURI/output'

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Decide which GPU to use to execute the code
# tf.config.experimental.VirtualDeviceConfiguration(memory_limit=12288)
# tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2000)

train_rows = 384  # 385, 525
train_cols = 520 # 385, 525
# train_rows = 160 # 385, 525
# train_cols = 160 # 385, 525

def append_model_params(cur_config):
    bands = len(cur_config[ProjTrainingParams.fields_names]) + len(cur_config[ProjTrainingParams.fields_names_obs])
    output_fields = len(cur_config[ProjTrainingParams.output_fields])
    model_config = {
        ModelParams.MODEL: AiModels.UNET_2D_SINGLE,
        ModelParams.DROPOUT: False,
        ModelParams.BATCH_NORMALIZATION: True,
        ModelParams.INPUT_SIZE: [train_rows, train_cols, bands],  # 4741632
        ModelParams.START_NUM_FILTERS: 16,
        ModelParams.NUMBER_LEVELS: 3,
        ModelParams.FILTER_SIZE: 5,
        ModelParams.OUTPUT_SIZE: output_fields
    }
    return {**cur_config, **model_config}


def appendFields(cur_config):
    model_config = {
        # ProjTrainingParams.fields_names:   ['u-vel.', 'v-vel.','temp', 'salin', 'thknss', 'srfhgt', 'mix_dpth'],
        ProjTrainingParams.fields_names:   [],
        # ProjTrainingParams.fields_names_obs: ['sst','sst', 'ssh_err', 'sst_err'],
        ProjTrainingParams.fields_names_obs: [],
        # ProjTrainingParams.fields_names_composite: ['diff_ssh',"topo","diff_sst"],
        ProjTrainingParams.fields_names_composite: [],
        # ProjTrainingParams.fields_names_var: ['tem', 'sal', 'ssh', 'mdt'],
        ProjTrainingParams.fields_names_var: [],
        # ProjTrainingParams.output_fields: ['srfhgt','temp', 'u-vel.', 'v-vel.', 'salin', 'thknss', 'srfhgt']
        ProjTrainingParams.output_fields: []
    }
    return {**cur_config, **model_config}


def get_training():
    cur_config = {
        # TrainingParams.output_folder: F"{join(_output_folder,'Training')}",
        TrainingParams.output_folder: F"{join(_output_folder,'TrainingPaper')}",
        TrainingParams.validation_percentage: .1,
        TrainingParams.test_percentage: .1,
        TrainingParams.file_name: 'RESULTS.csv',

        TrainingParams.evaluation_metrics: [mymetrics.only_ocean_mse, mymetrics.mse],  # Metrics to show in tensor flow in the training
        TrainingParams.loss_function: mymetrics.only_ocean_mse,  # Loss function to use for the learning
        # TrainingParams.evaluation_metrics: [losses.mse],  # Metrics to show in tensor flow in the training
        # TrainingParams.loss_function: losses.mse,  # Loss function to use for the learning

        TrainingParams.optimizer: Adam(lr=0.001),  # Default values lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None,
        TrainingParams.batch_size: 10, # READ In this case it is not a common batch size. It indicates the number of images to read from the same file
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
        ProjTrainingParams.perc_ocean: 0,
        ProjTrainingParams.network_type: NetworkTypes.UNET,
        ProjTrainingParams.output_folder_summary_models:  F"{join(_output_folder,'SUMMARY')}",
    }
    return append_model_params(appendFields(cur_config))


def get_prediction_params():
    weights_folder = join(_output_folder,'Training', _run_name, 'models')
    cur_config = {
        TrainingParams.config_name: _run_name,
        PredictionParams.input_folder: _preproc_folder,
        PredictionParams.output_folder: F"{join(_output_folder,'Prediction2002_2006')}",
        PredictionParams.output_imgs_folder: F"{join(_output_folder,'Prediction2002_2006','imgs')}",
        PredictionParams.show_imgs: False,
        PredictionParams.model_weights_file: join(weights_folder, "0002_GoM2D_STDNORM_PERCOCEAN_0_NET_2DUNET_srfhgt_ssh-ssh-err-sst-sst-err_No-STD_OUT_SRFHGT_384x520_2021_10_24_18_05-epoch-64-loss-0.00018359.hdf5"),
        PredictionParams.metrics: mymetrics.only_ocean_mse,
    }

    return {**append_model_params(appendFields(cur_config)), **get_training()}

