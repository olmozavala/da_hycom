# %%
import matplotlib.pyplot as plt
import os
from os.path import join
import pandas as pd
from ExtraUtils.NamesManipulation import *
import numpy as np

# This code is the one that makes all the bar plots varying the options selected in each training

fontsize = 20
colors = ['pink', 'lightblue', 'lightgreen','lightyellow','lightgrey']

def makeScatter(c_summary, group_field, xlabel, output_file, title_txt="", units=""):
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
    ax.set_ylabel(F"RMSE {units}", fontsize=fontsize)
    if title_txt == "":
        ax.set_title(F"RMSE by {group_field} for the test set", fontsize=fontsize)
    else:
        ax.set_title(F"RMSE by {title_txt} for the test set", fontsize=fontsize)
    # plt.xticks(range(len(names)),names)
    print("Plotting....")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()
    plt.close()

imgs_prediction_folder = "/data/HYCOM/DA_HYCOM_TSIS/Prediction2002_2006/imgs"
# imgs_prediction_folder = "/data/HYCOM/DA_HYCOM_TSIS/PredictionPaper/imgs"
summary_folder = "/data/HYCOM/DA_HYCOM_TSIS/SUMMARY/"
# imgs_prediction_folder = "/home/olmozavala/DAN_HYCOM/OUTPUT/Prediction2002_2006/imgs/"
# summary_folder = "/home/olmozavala/DAN_HYCOM/OUTPUT/SUMMARY/"
summary_file_name = "summary.csv"
# summary_file_name = "Summary_Only_Best"

all_folders = [ name for name in os.listdir(imgs_prediction_folder) if os.path.isdir(os.path.join(imgs_prediction_folder, name)) ]
all_folders.sort()

For_2002_2006 = False  # Indicates if we are doing it for 2002 and 2006
generate_scatter_by_model = False # These are the individual scatter plots for each model

# Define the summary_file
summary_file = join(summary_folder,summary_file_name)

output_file = join(summary_folder, f"RMS_{summary_file_name}.csv")

summary_df = pd.read_csv(summary_file)
training_folder = '/'.join(np.array(os.path.dirname(summary_df.iloc[0]["Path"]).split("/"))[:-1])

# train_id = 291
# val_id = 291+37
# For all test plots

# Makes a scatter plot k
for c_model in all_folders:
    print(F"------------------- {c_model} -----------------------")
    c_file = join(imgs_prediction_folder, c_model, "Global_RMSE_and_times.csv")
    # Match summary with model
    cur_sum_model = summary_df["Path"] == join(training_folder,c_model,"models")

    split_model = c_model.split("_")
    # model_title = F"{split_model[0]} %{int(split_model[4])/10} {split_model[6]}_{split_model[7]} {split_model[8]} {split_model[12]}"
    model_title = F"{split_model[0]} %{int(split_model[4])/10} {split_model[6]}_{split_model[7]} {split_model[8]} {split_model[11]}"
    try:
        network = getNetworkTypeTxt(c_model)
        output_field = getOutputFieldsTxt(c_model)
        if os.path.exists(c_file):
            df = pd.read_csv(c_file)

            if For_2002_2006:
                rms_mean = df["rmse"]
                dates_str = df["File"].values
            else:
                # Check if the first column of the dataframe is a number
                if isinstance(df["rmse"][0], (int, float)):
                    rms_mean = df["rmse"].values
                else:
                    rms_mean = np.array([float(x[1:-1].split(' ')[0]) for x in df["rmse"]]) # Used in paper
                dates_str = [int(x.split("_")[-2])for x in df["File"]]

            if np.any(np.isnan(rms_mean)):
                print(F"Failed for {c_file} {e}")
                continue

            if generate_scatter_by_model: # Plot part
                if For_2002_2006:
                    months = ['F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N']
                    xticks_labels = ['2002'] + months + ['2006'] + months
                    xticks_pos = np.linspace(0, len(dates_str) - 1, len(xticks_labels))

                plt.figure(figsize=(10,5))
                plt.scatter(dates_str, rms_mean)
                # Only used if you want to plot where the training and validation cuts are
                # plt.axvline(x=dates_str[train_id], c='g')
                # plt.axvline(x=dates_str[val_id], c='r')
                # plt.ylim([0.001,0.03])
                plt.ylabel('RMSE m')
                plt.xlabel(F'Day of the year')
                if For_2002_2006:
                    plt.title(F"{model_title}\nRMSE of SSH prediction of years 2002 and 2006: {rms_mean.mean():0.4f} ")
                else:
                    # plt.title(F"RMSE of SSH increment prediction by dataset: \n training: {rms_mean.iloc[:train_id].mean():0.3f}  validation:{rms_mean.iloc[train_id:val_id].mean():0.3f} test:{rms_mean.iloc[val_id:].mean():0.3f} ")
                    plt.title(F"Mean RMSE of SSH increment prediction test:{rms_mean.mean():0.3f} m")
                print(join(imgs_prediction_folder, F"{c_model}_RMSE.png"))
                if For_2002_2006:
                    plt.xticks(xticks_pos, labels=xticks_labels, rotation=0)
                plt.savefig(join(imgs_prediction_folder, F"{c_model}_RMSE.png") ,bbox_inches='tight', dpi=300)
                plt.show()
            plt.close()
        summary_df.loc[cur_sum_model, "RMSE"] = rms_mean.mean()
    except Exception as e:
        print(F"Failed for {c_file} {e}")

summary_df.to_csv(output_file)

NET = "Network Type"
OUT = "Output vars"
IN  = "Input vars"
LOSS  = "Loss value"
PERCOCEAN = "PercOcean"
print("Done reading data")

# %%
# Plot summary by
output_folder = "/home/olmozavala/Dropbox/Apps/Overleaf/CNN_DA/EXTRA/imgs"
# output_folder = summary_folder
# ========= Compare Network type ======
c_summary = summary_df[np.logical_and((summary_df[IN] == "ssh").values, (summary_df[OUT] == "SRFHGT").values)]
c_summary = c_summary[c_summary["BBOX"] == "384x520"]  # Only ifull domain
c_summary = c_summary[c_summary[PERCOCEAN] <= 0.1]  # Only PercOcean 0.0
# def makeScatter(c_summary, group_field, xlabel, output_file, title_txt="", units=""):
makeScatter(c_summary, NET, "Network Architecture", join(output_folder,"By_Network_Type_Scatter_TestSet.png"),
            title_txt="Network Architecture",
            units="(meters)")
# %%
# # ========= Compare BBOX ======
c_summary = summary_df[np.logical_and((summary_df[IN] == "ssh").values, (summary_df[OUT] == "SRFHGT").values)]
c_summary = c_summary[c_summary[NET] == "2DUNET"]   # Only UNet
c_summary = c_summary[c_summary[PERCOCEAN] == 0.0]  # Only PercOcean 0.0
makeScatter(c_summary, "BBOX", "Window Size", join(output_folder,"By_bbox_Type_Scatter_TestSet.png"),
            title_txt="Window Size",
            units="(meters)")

# ========= Compare PercOcean ======
c_summary = summary_df[np.logical_and((summary_df[IN] == "ssh").values, (summary_df[OUT] == "SRFHGT").values)]
c_summary = c_summary[c_summary["BBOX"] == "160x160"]  # Only 160x160
c_summary = c_summary[c_summary[NET] == "2DUNET"]   # Only UNet
makeScatter(c_summary, PERCOCEAN, "Percentage of Ocean", join(output_folder,"By_PercOcean_Type_Scatter_TestSet.png"), "percentage of ocean", units="(meters)")

# ========= Compare Inputs type ======
c_summary = summary_df[np.logical_and(summary_df[NET] == "2DUNET", (summary_df[OUT] == "SRFHGT").values)]
c_summary = c_summary[c_summary["BBOX"] == "384x520"]  # Only 160x160
c_summary = c_summary[c_summary[PERCOCEAN] == 0.0]  # Only PercOcean 0.0
makeScatter(c_summary, IN, "Input fields", join(output_folder,"By_Input_Type_Scatter_TestSet.png"), "input fields", units="(meters)")

# ========= Compare Outputs type ======
c_summary = summary_df[np.logical_and(summary_df[NET] == "2DUNET", (summary_df[IN] == "ssh-sst").values)]
c_summary = c_summary[c_summary["BBOX"] == "384x520"]  # Only 160x160
c_summary = c_summary[c_summary[PERCOCEAN] == 0.0]  # Only PercOcean 0.0
makeScatter(c_summary, OUT, "Output fields", join(output_folder,"By_Output_Type_Scatter_TestSet.png"), "output fields",
            units="(meters and degrees)")
# %%
