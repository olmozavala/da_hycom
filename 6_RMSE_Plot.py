import matplotlib.pyplot as plt
import os
from os.path import join
import pandas as pd
from ExtraUtils.NamesManipulation import *
import numpy as np

def makeScatter(c_summary, group_field, xlabel, output_file):
    if c_summary.empty:
        return

    RMSE = "RMSE"
    names = []
    data = []
    fig, ax = plt.subplots(figsize=(10,6))
    for i, grp in enumerate(c_summary.groupby(group_field)):
        names.append(grp[0])
        c_data = grp[1][RMSE].values
        data.append(c_data)
        plt.scatter(np.ones(len(c_data))*i, c_data, label=grp[0])

    plt.legend(loc="best")
    # bp = plt.boxplot(data, labels=names, patch_artist=True, meanline=True, showmeans=True)
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("RMSE")
    ax.set_title(F"RMSE by {group_field} Type ")
    plt.savefig(output_file)
    plt.show()

imgs_prediction_folder = "/data/HYCOM/DA_HYCOM_TSIS/Prediction/imgs"

all_folders = [ name for name in os.listdir(imgs_prediction_folder) if os.path.isdir(os.path.join(imgs_prediction_folder, name)) ]
all_folders.sort()

# Define the summary_file
summary_file = "/data/HYCOM/DA_HYCOM_TSIS/SUMMARY/summary.csv"
summary_df = pd.read_csv(summary_file)
training_folder = '/'.join(np.array(os.path.dirname(summary_df.iloc[0]["Path"]).split("/"))[:-1])

# train_id = 291
# val_id = 291+37
# For all test plots

for c_model in all_folders:
    print(F"------------------- {c_model} -----------------------")
    c_file = join(imgs_prediction_folder, c_model, "Global_RMSE_and_times.csv")
    # Match summary with model
    cur_sum_model = summary_df["Path"] == join(training_folder,c_model,"models")

    split_model = c_model.split("_")
    model_title = F"{split_model[0]} %{int(split_model[4])/10} {split_model[6]}_{split_model[7]} {split_model[8]} {split_model[12]}"
    try:
        network = getNetworkTypeTxt(c_model)
        output_field = getOutputFieldsTxt(c_model)
        if os.path.exists(c_file):
            df = pd.read_csv(c_file)
            rms_mean = df["rmse"]
            year_day = [int(x.split("_")[-2])for x in df["File"]]
            plt.figure(figsize=(10,5))
            plt.scatter(year_day, rms_mean)
            # Only used if you want to plot where the trainging and validation cuts are
            # plt.axvline(x=year_day[train_id], c='g')
            # plt.axvline(x=year_day[val_id], c='r')
            plt.ylim([0.001,0.03])
            plt.ylabel('Meters')
            plt.xlabel(F'Day of the year {}')
            # plt.title(F"RMSE of SSH prediction by dataset: \n training: {rms_mean.iloc[:train_id].mean():0.3f}  validation:{rms_mean.iloc[train_id:val_id].mean():0.3f} test:{rms_mean.iloc[val_id:].mean():0.3f} ")
            plt.title(F"{model_title}\nRMSE of SSH prediction of test dataset: {rms_mean.mean():0.4f} ")
            print(join(imgs_prediction_folder, F"{c_model}_RMSE.png"))
            plt.savefig(join(imgs_prediction_folder, F"{c_model}_RMSE.png") ,bbox_inches='tight', dpi=300)
            # plt.show()
            plt.close()
        summary_df.loc[cur_sum_model, "RMSE"] = df["rmse"].mean(0)
    except Exception as e:
        print(F"Failed for {c_file} {e}")

summary_df.to_csv(summary_file)

NET = "Network Type"
OUT = "Output vars"
IN  = "Input vars"
LOSS  = "Loss value"
PERCOCEAN = "PercOcean"
# Plot summary by
# ========= Compare Network type ======
c_summary = summary_df[np.logical_and((summary_df[IN] == "No-STD").values, (summary_df[OUT] == "SRFHGT").values)]
c_summary = c_summary[c_summary["BBOX"] == "160x160"]  # Only 160x160
c_summary = c_summary[c_summary[PERCOCEAN] == 0.0]  # Only PercOcean 0.0
makeScatter(c_summary, NET, "Network type", join(imgs_prediction_folder,"By_Network_Type_Scatter.png"))

# ========= Compare PercOcean ======
c_summary = summary_df[np.logical_and((summary_df[IN] == "No-STD").values, (summary_df[OUT] == "SRFHGT").values)]
c_summary = c_summary[c_summary["BBOX"] == "160x160"]  # Only 160x160
c_summary = c_summary[c_summary[NET] == "2DUNET"]   # Only UNet
makeScatter(c_summary, PERCOCEAN, "Perc Ocean", join(imgs_prediction_folder,"By_PercOcean_Type_Scatter.png"))

# ========= Compare BBOX ======
c_summary = summary_df[np.logical_and((summary_df[IN] == "No-STD").values, (summary_df[OUT] == "SRFHGT").values)]
c_summary = c_summary[c_summary[NET] == "2DUNET"]   # Only UNet
c_summary = c_summary[c_summary[PERCOCEAN] == 0.0]  # Only PercOcean 0.0
makeScatter(c_summary, "BBOX", "BBOX size", join(imgs_prediction_folder,"By_bbox_Type_Scatter.png"))
##

