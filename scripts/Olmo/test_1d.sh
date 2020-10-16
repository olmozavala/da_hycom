#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:/home/olmozavala/Dropbox/MyProjects/OZ_LIB/image_visualization"
export PYTHONPATH="${PYTHONPATH}:/home/olmozavala/Dropbox/MyProjects/OZ_LIB/image_preprocessing"

RUN_FOLDER='Test'
SRC_PATH='/home/olmozavala/Dropbox/MyProjects/OZ_LIB/AI_Template'
MAIN_CONFIG="${SRC_PATH}/config"

#echo '############################ Copying configuration file ############################ '
#cp Config_Tests.py $MAIN_CONFIG/MainConfig.py


echo '############################ Training ############################ '
python $SRC_PATH/Train_Time_Series.py
