# %%
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from os.path import join
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

from config.MainConfig_2D import get_training
from ai_common.constants.AI_params import TrainingParams, ModelParams
from constants_proj.AI_proj_params import ProjTrainingParams

import sys
sys.path.append("eoas_pyutils/")
from io_utils.io_common import create_folder

from ExtraUtils.NamesManipulation import *


# This code is used to generate a summary of the models
# that have been tested. Grouping by each modified parameter
def changeDecimal(name):
    name = name.replace("SimpleCNN_2", "SimpleCNN_02")
    name = name.replace("SimpleCNN_4", "SimpleCNN_04")
    name = name.replace("SimpleCNN_8", "SimpleCNN_08")
    return name

def buildSummary(name, loss, path):
    model = [name,
             changeDecimal(getNetworkTypeTxt(name)),
             getBBOXandText(name)[2],
             getPercOcean(name)[1],
             getInputFieldsTxt(name),
             getOutputFieldsTxt(name),
             loss, path, getId(name)]
    return model

def buildDF(summary):
    df = {
        "ID": [x[8] for x in summary],
        "Network Type": [x[1] for x in summary],
        "BBOX": [x[2] for x in summary],
        "PercOcean": [x[3] for x in summary],
        "Input vars":[x[4] for x in summary],
        "Output vars": [x[5] for x in summary],
        "Loss value": [x[6] for x in summary],
        "Name": [x[0] for x in summary],
        "Path": [x[7] for x in summary],
    }
    return pd.DataFrame.from_dict(df)

def fixNames(trained_models_folder):
    print("================ Fixing files names ========================")
    from_txt = "_NET_NET_"
    to_txt = "_NET_"

    # First fix all the directories
    for root, dirs, files in os.walk(trained_models_folder):
        for name in dirs:
            if name.find(from_txt) != -1:
                old_name = join(root, name)
                new_name = join(root, name.replace(from_txt, to_txt))
                print(F"From {old_name} \nTo   {new_name} \n")
                os.rename(old_name, new_name)

    # Then all the files
    for root, dirs, files in os.walk(trained_models_folder):
        for name in files:
            if name.find(from_txt) != -1:
                old_name = join(root, name)
                new_name = join(root, name.replace(from_txt, to_txt))
                print(F"{old_name} \n {new_name} \n")
                os.rename(old_name, new_name)

    # filter_file = "GoM2D"
    # for root, dirs, files in os.walk(trained_models_folder):
    #     for name in dirs:
    #         if name.find(filter_file) != -1:
    #             old_name = join(root, name)
    #             new_name = join(root, changeDecimal(name))
    #             # print(F"From {old_name} \nTo   {new_name} \n")
    #             # os.rename(old_name, new_name)
    #
    # # Then all the files
    # for root, dirs, files in os.walk(trained_models_folder):
    #     for dirname in dirs:
    #         print(dirname)
    #     for name in files:
    #         if name.find(from_txt) != -1:
    #             old_name = join(root, name)
    #             new_name = join(root, changeDecimal(name))
    #             print(F"{old_name} \n {new_name} \n")
    #             # os.rename(old_name, new_name)

fontsize = 20
colors = ['pink', 'lightblue', 'lightgreen','lightyellow','lightgrey']

def makeScatterTest(c_summary, group_field, xlabel, output_file, title_txt="", units=""):
    if c_summary.empty:
        return

    # RMSE = "RMSE"
    RMSE = "Loss value"
    names = []
    data = []
    fig, ax = plt.subplots(figsize=(10,6))
    for i, grp in enumerate(c_summary.groupby(group_field)):
        names.append(grp[0])
        c_data = grp[1][RMSE].values
        data.append(c_data)
        # plt.scatter(np.ones(len(c_data))*i, c_data, label=grp[0])

    if group_field == "Network Type":  # Horrible hack to correct the order of the Unet
        n = len(names)
        names.insert(n, names.pop(0))
        data.insert(n, data.pop(0))

    if group_field == "PercOcean":  # Horrible hack to correct the names for percentage of ocean
        names = [f"{int(x*100)}%" for x in names]

    if group_field == "Input vars":  # Horrible hack to correct the names for input fields
        names = ['SSH', 'SSH, SSH-ERR, SST, SST-ERR', 'SSH, SST']

    if group_field == "Output vars":  # Horrible hack to correct the names for outpu fields
        names = ['SSH', 'SSH, SST', 'SST']

    # Plotting with boxplot
    # plt.boxplot(data,  labels=names, patch_artist=True)
    # plt.legend(loc="best")
    # bp = ax.boxplot(data, labels=names, patch_artist=True)

    boxprops = dict(linestyle='-', linewidth=0.5, color='grey')
    bp = ax.boxplot(data, labels=names, patch_artist=True, showmeans=True, boxprops=boxprops)
    for i, patch in enumerate(bp['boxes']):
        # patch.set_facecolor(colors[i])
        patch.set_facecolor('lightgrey')

    plt.tick_params(axis='x', which='major', labelsize=fontsize-7)
    plt.tick_params(axis='y', which='major', labelsize=fontsize-7)
    # plt.xticks(rotation=-5)
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(F"RMSE (meters)", fontsize=fontsize)
    if title_txt == "":
        ax.set_title(F"RMSE by {group_field} with Lat&Lon inputs for the test set", fontsize=fontsize)
    else:
        ax.set_title(F"RMSE by {title_txt} for the test set", fontsize=fontsize)
    # plt.xticks(range(len(names)),names)
    print("Plotting....")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()
    plt.close()

def makeScatter(c_summary, group_field, xlabel, output_file):
    """
    Makes a scatter plot from a dataframe grouped by the specified "group_field"
    :param c_summary:
    :param group_field:
    :param xlabel:
    :param output_file:
    :return:
    """
    LOSS  = "Loss value"
    names = []
    data = []
    fig, ax = plt.subplots(figsize=(10,6))
    for i, grp in enumerate(c_summary.groupby(group_field)):
        names.append(grp[0])
        c_data = grp[1][LOSS].values
        data.append(c_data)
        plt.scatter(np.ones(len(c_data))*i, c_data, label=grp[0])

    plt.legend(loc="best")
    # bp = plt.boxplot(data, labels=names, patch_artist=True, meanline=True, showmeans=True)
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Validation Loss (MSE)")
    ax.set_title(F"Validation Loss by {group_field} Type ")
    plt.savefig(join(output_folder,output_file))
    plt.show()

# %%
NET = "Network Type"
OUT = "Output vars"
IN  = "Input vars"
LOSS  = "Loss value"
PERCOCEAN = "PercOcean"

# Read folders for all the experiments
config = get_training()
trained_models_folder = config[TrainingParams.output_folder]
output_folder = config[ProjTrainingParams.output_folder_summary_models]
create_folder(output_folder)

# fixNames("/data/HYCOM/DA_HYCOM_TSIS/Training")
# exit()

all_folders = os.listdir(trained_models_folder)
all_folders.sort()

summary = []

# Iterate over all the experiments
for experiment in all_folders:
    print(F"Working with experiment: {experiment}")
    if experiment == "training_imgs":
        break
    all_models = os.listdir(join(trained_models_folder, experiment , "models"))
    min_loss = 100000.0
    best_model = {}
    # Iterate over the saved models for each experiment and obtain the best of them
    for model in all_models:
        loss = float((model.split("-")[-1]).replace(".hdf5",""))
        if loss < min_loss:
            min_loss = loss
            best_model = buildSummary(model, np.around(min_loss,5), join(trained_models_folder, experiment, "models"))
    # Verify that model object is not an empty dictionary
    if best_model:
        summary.append(best_model)
summary = buildDF(summary)
print(summary)

summary.to_csv(join(output_folder,"summary.csv"))

# %%
def_bbox = "384x520"
def_in = "ssh-ssh-err"
def_out = "SRFHGT"
def_perc_ocean = "0.0"

# %% ========= Compare Network type ======
c_summary = summary[np.logical_and((summary[IN] == def_in).values, (summary[OUT] == def_out).values)]
c_summary = c_summary[c_summary["BBOX"] == def_bbox]
c_summary = c_summary[c_summary[PERCOCEAN] == def_perc_ocean]  # Only PercOcean 0.0
makeScatter(c_summary, NET, "Network type", "By_Network_Type_Scatter.png")

# %% ========= Compare PercOcean ======
c_summary = summary[np.logical_and((summary[IN] == def_in).values, (summary[OUT] == def_out).values)]
c_summary = c_summary[c_summary["BBOX"] == "160x160"]  # Only 160x160
c_summary = c_summary[c_summary[NET] == "2DUNET"]   # Only UNet
makeScatter(c_summary, PERCOCEAN, "Perc Ocean", "By_PercOcean_Type_Scatter.png")

# %% ========= Compare BBOX ======
c_summary = summary[np.logical_and((summary[IN] == def_in).values, (summary[OUT] == def_out).values)]
c_summary = c_summary[c_summary[NET] == "2DUNET"]   # Only UNet
c_summary = c_summary[c_summary[PERCOCEAN] == def_perc_ocean]  # Only PercOcean 0.0
print(f"Total number of models for the comparison of BBOX size per group is: {len(c_summary)}")
makeScatter(c_summary, "BBOX", "BBOX size", "By_bbox_Type_Scatter.png")
# makeScatterTest(c_summary, "BBOX", "Window size", "By_bbox_Type_Scatter_TestSet.png")

# %% ========= Compare OBS INPUT ======
c_summary = summary[summary[OUT] == def_out]
c_summary = c_summary[c_summary[NET] == "2DUNET"]
c_summary = c_summary[c_summary[PERCOCEAN] == def_perc_ocean]  # Only PercOcean 0.0
c_summary = c_summary[c_summary["BBOX"] == def_bbox]  # Only 160x160
makeScatter(c_summary, IN, "BBOX size", "By_Inputs_Scatter.png")


# %% ========= Compare OBS Output ======
c_summary = summary[summary[NET] == "2DUNET"]
c_summary = c_summary[c_summary[PERCOCEAN] == def_perc_ocean]  # Only PercOcean 0.0
c_summary = c_summary[c_summary["BBOX"] == def_bbox]  # Only 160x160
makeScatter(c_summary, OUT, "Ouptut type", "By_Output_Scatter.png")
print("Done!")
# %%
