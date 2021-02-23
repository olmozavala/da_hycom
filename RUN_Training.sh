#!/bin/bash

PYTHONPATH=""
PYTHONPATH=$PYTHONPATH:/home/olmozavala/Dropbox/MyProjects/OZ_LIB/image_visualization
PYTHONPATH=$PYTHONPATH:/home/olmozavala/Dropbox/MyProjects/OZ_LIB/eoas_preprocessing
export PYTHONPATH=$PYTHONPATH:/home/olmozavala/Dropbox/MyProjects/OZ_LIB/AI_Common

echo $PYTHONPATH
SRC_PATH='/home/olmozavala/Dropbox/MyProjects/EOAS/COAPS/MURI_AI_Ocean/Data_Assimilation/HYCOM-TSIS'

echo '############################ Training ############################ '
python $SRC_PATH/3_Train_2D.py > CurrentRun.txt &

