import os
from os.path import join
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

from config.MainConfig import get_training_2d
from constants.AI_params import TrainingParams, ModelParams
from constants_proj.AI_proj_params import ProjTrainingParams
from img_viz.common import create_folder

# This code is used to generate a summary of the models
# that have been tested. Grouping by each modified parameter

def getNetworkType(name):
    if "simple_cnn" in name:
        return "Simple_CNN"
    if "unet" in name:
        return "UNET"
    if "unet_multistream" in name:
        return "UNET_MultiStream"

    return "Unknown"

def getInputFields(name):
    if "Input_All_with_Obs_No_SSH_NO_LATLON".lower() in name:
        # return "ALL no SSH, no LATLON ( temp, salin, u-vel., v-vel., sst, sss, ssh)"
        return "ALLNoSSHnoLATLON"
    if "all_intput_with_latlon" in name:
        # return "ALL with LATLON ( temp, srfhgt, salin, u-vel., v-vel., LAT, LON ,sst, sss, ssh)"
        return "ALLwithLATLON"
    if "all_input" in name or "input_all" in name:
        # return "ALL ( temp, srfhgt, salin, u-vel., v-vel., sst, sss, ssh)"
        return "ALL"
    if "input_temp" in name:
        return "Temp"
    if "onlytemp" in name:
        return "Temp"

    return "Unknown"

def getOutputFields(name):
    if "output_only_temp" in name or "onlytemp" in name:
        return "Temp"
    if "output_only_ssh" in name or "onlyssh" in name:
        return "SSH"
    if "all_output" in name or "output_all" in name:
        return "All"
    return "Unknown"

def getBBOX(name):
    if "160x160" in name:
        return "160x160"
    if "80x80" in name:
        return "80x80"
    return "Unknown"

def landperc(name):
    if "no_land" in name.lower():
        return "No Land"
    return "Unknown"

def buildSummary(orig_name, loss, path):
    name = orig_name.lower()
    model = [orig_name,  getNetworkType(name), getBBOX(name), getInputFields(name), getOutputFields(name), loss, path]
    return model

def buildDF(summary):
    df = {
        "Network Type": [x[1] for x in summary],
        "BBOX": [x[2] for x in summary],
        "Input vars":[x[3] for x in summary],
        "Output vars": [x[4] for x in summary],
        "Loss value": [x[5] for x in summary],
        "Name": [x[0] for x in summary],
        "Path": [x[6] for x in summary],
    }
    return pd.DataFrame.from_dict(df)

if __name__ == '__main__':

    # Read folders for all the experiments
    config = get_training_2d()
    trained_models_folder = config[TrainingParams.output_folder]
    output_folder = config[ProjTrainingParams.output_folder_summary_models]
    create_folder(output_folder)

    all_folders = os.listdir(trained_models_folder)
    print(all_folders)

    summary = []
    # Iterate over all the experiments
    for experiment in all_folders:
        all_models = os.listdir(join(trained_models_folder, experiment , "models"))
        min_loss = 1.0
        best_model = {}
        # Iterate over the saved models for each experiment and obtain the best of them
        for model in all_models:
            loss = float((model.split("-")[-1]).replace(".hdf5",""))
            if loss < min_loss:
                min_loss = loss
                best_model = buildSummary(model, np.around(min_loss,5), join(trained_models_folder, experiment, "models", model))
        summary.append(best_model)
    summary = buildDF(summary)
    print(summary)

    for col in ["Network Type", "BBOX", "Input vars", "Output vars"]:
        for grp in summary.groupby(col):
            plt.scatter(range(len(grp[1])), grp[1]["Loss value"].values, label=grp[0])
        plt.legend(bbox_to_anchor=(1.1, 1.05))
        plt.title(F"Grouped by {col}")
        plt.savefig(join(output_folder,F"{col.replace(' ','_')}.png"))
        plt.show()

    print("Done!")
summary.to_csv(join(output_folder,"summary.csv"))