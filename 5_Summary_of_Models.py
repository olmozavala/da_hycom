import os
from os.path import join
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

from config.MainConfig_2D import get_training
from constants.AI_params import TrainingParams, ModelParams
from constants_proj.AI_proj_params import ProjTrainingParams
from img_viz.common import create_folder

from ExtraUtils.NamesManipulation import *


# This code is used to generate a summary of the models
# that have been tested. Grouping by each modified parameter
def changeDecimal(name):
    name = name.replace("SimpleCNN_2", "SimpleCNN_02")
    name = name.replace("SimpleCNN_4", "SimpleCNN_04")
    name = name.replace("SimpleCNN_8", "SimpleCNN_08")
    return name

def buildSummary(name, loss, path):
    model = [name, changeDecimal(getNetworkTypeTxt(name)), getBBOX(name), getInputFieldsTxt(name), getOutputFieldsTxt(name), loss, path, getId(name)]
    return model

def buildDF(summary):
    df = {
        "ID": [x[7] for x in summary],
        "Network Type": [x[1] for x in summary],
        "BBOX": [x[2] for x in summary],
        "Input vars":[x[3] for x in summary],
        "Output vars": [x[4] for x in summary],
        "Loss value": [x[5] for x in summary],
        "Name": [x[0] for x in summary],
        "Path": [x[6] for x in summary],
    }
    return pd.DataFrame.from_dict(df)

def fixNames(trained_models_folder):
    print("================ Fixing files names ========================")
    from_txt = "IN_ALL"
    to_txt = "IN_Yes-STD"
    # First fix all the directories
    # for root, dirs, files in os.walk(trained_models_folder):
    #     for name in dirs:
    #         if name.find(from_txt) != -1:
    #             old_name = join(root, name)
    #             new_name = join(root, name.replace(from_txt, to_txt))
    #             print(F"From {old_name} \nTo   {new_name} \n")
    #             os.rename(old_name, new_name)
    #
    # # Then all the files
    # for root, dirs, files in os.walk(trained_models_folder):
    #     for name in files:
    #         if name.find(from_txt) != -1:
    #             old_name = join(root, name)
    #             new_name = join(root, name.replace(from_txt, to_txt))
    #             print(F"{old_name} \n {new_name} \n")
    #             os.rename(old_name, new_name)

    filter_file = "SimpleCNN"
    for root, dirs, files in os.walk(trained_models_folder):
        for name in dirs:
            if name.find(filter_file) != -1:
                old_name = join(root, name)
                new_name = join(root, changeDecimal(name))
                # print(F"From {old_name} \nTo   {new_name} \n")
                # os.rename(old_name, new_name)

    # Then all the files
    for root, dirs, files in os.walk(trained_models_folder):
        for dirname in dirs:
            print(dirname)
        for name in files:
            if name.find(from_txt) != -1:
                old_name = join(root, name)
                new_name = join(root, changeDecimal(name))
                print(F"{old_name} \n {new_name} \n")
                # os.rename(old_name, new_name)

if __name__ == '__main__':

    NET = "Network Type"
    OUT = "Output vars"
    IN  = "Input vars"
    LOSS  = "Loss value"

    # Read folders for all the experiments
    config = get_training()
    trained_models_folder = config[TrainingParams.output_folder]
    output_folder = config[ProjTrainingParams.output_folder_summary_models]
    create_folder(output_folder)

    # fixNames("/data/HYCOM/DA_HYCOM_TSIS/Training")
    # exit()

    all_folders = os.listdir(trained_models_folder)
    all_folders.sort()
    print(all_folders)

    summary = []

    # Iterate over all the experiments
    for experiment in all_folders:
        all_models = os.listdir(join(trained_models_folder, experiment , "models"))
        min_loss = 100000.0
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

    summary.to_csv(join(output_folder,"summary.csv"))

    # ========= Compare Network type ======
    # data_novar = summary[summary[IN] == "No-STD"]  # All no STD data as input
    # data_novar_srfhgt = data_novar[data_novar[OUT] == "SRFHGT"]  # Only SSH data
    #
    # names = []
    # data = []
    # fig, ax = plt.subplots(figsize=(10,6))
    # for i, grp in enumerate(data_novar_srfhgt.groupby(NET)):
    #     names.append(grp[0])
    #     c_data = grp[1][LOSS].values
    #     data.append(c_data)
    #     plt.scatter(np.ones(len(c_data))*i, c_data, label=grp[0])
    #
    # plt.legend(loc="best")
    # # bp = plt.boxplot(data, labels=names, patch_artist=True, meanline=True, showmeans=True)
    # ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    # ax.set_xlabel("Network type")
    # ax.set_ylabel("Validation Loss (MSE)")
    # ax.set_title("Validation Loss by Network Type (SSH)")
    # plt.savefig(join(output_folder,F"By_Network_Type_Scatter.png"))
    # plt.show()

    # ========= Compare By network inputs ======
    fig, ax = plt.subplots(figsize=(10,6))
    for j, major_grp in enumerate(summary.groupby(OUT)):
        out_name = major_grp[0]
        data_out = summary[summary[OUT] == out_name]  # All srfhgt data
        names = []
        data = []
        for i, grp in enumerate(data_out.groupby(IN)):
            names.append(grp[0])
            c_data = grp[1][LOSS].values
            data.append(c_data)
            # plt.scatter(np.ones(len(c_data))*i, c_data, label=grp[0])

        # plt.legend(loc="best")
        bp = plt.boxplot(data, labels=names, patch_artist=True, meanline=True, showmeans=True)
        ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
        ax.set_xlabel("Input variables")
        ax.set_ylabel("Validation Loss (MSE)")
        plt.title(F"Validation Loss by output: {out_name}")
        plt.savefig(join(output_folder,F"{out_name}_out.png"))
        plt.show()

    # # ========= Compare by Output field ======
    # data_novar_unet = data_novar[data_novar[NET] == "UNET"]  # All srfhgt data
    #
    # names = []
    # data = []
    # fig, ax = plt.subplots(figsize=(10,6))
    # for i, grp in enumerate(data_novar_unet.groupby(OUT)):
    #     names.append(grp[0])
    #     c_data = grp[1][LOSS].values
    #     data.append(c_data)
    #     # plt.scatter(np.ones(len(c_data))*i, c_data, label=grp[0])
    #
    # # plt.legend(loc='best')
    # bp = plt.boxplot(data, labels=names, patch_artist=True, meanline=True, showmeans=True)
    # ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    # ax.set_xlabel("Output field")
    # ax.set_ylabel("Validation Loss (MSE)")
    # ax.set_title("Validation Loss by output field (UNET)")
    # plt.savefig(join(output_folder,F"By_OutField_Type.png"))
    # plt.show()
    # #
    # # # ========= Var vs no Var======
    # names = []
    # data = []
    # fig, ax = plt.subplots(figsize=(10,6))
    # for i, grp in enumerate(summary.groupby(IN)):
    #     if grp[0].find("Yes") != -1:
    #         names.append(F"{grp[0]} (SST, SSS, SSH)")
    #     else:
    #         names.append(F"{grp[0]} ")
    #     data.append(grp[1][LOSS].values)
    #     plt.scatter(np.ones(len(grp[1]))*i, grp[1][LOSS].values, label=grp[0])
    #
    # plt.legend(loc="center")
    # # bp = plt.boxplot(data, labels=names, patch_artist=True, meanline=True, showmeans=True)
    # ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    # ax.set_xlabel("Input field")
    # ax.set_ylabel("Validation Loss (MSE)")
    # ax.set_title("Validation Loss by input fields")
    # plt.savefig(join(output_folder,F"By_InputField_Type_Scatter.png"))
    # plt.show()

    print("Done!")
