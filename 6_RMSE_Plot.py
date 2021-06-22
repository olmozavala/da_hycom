import matplotlib.pyplot as plt
import os
from os.path import join
import pandas as pd
from ExtraUtils.NamesManipulation import *

imgs_prediction_folder = "/data/HYCOM/DA_HYCOM_TSIS/Prediction/imgs"

all_folders = os.listdir(imgs_prediction_folder)
all_folders.sort()

for c_model in all_folders:
    c_file = join(imgs_prediction_folder, c_model, "Global_RMSE_and_times.csv")
    try:
        network = getNetworkTypeTxt(c_model)
        output_field = getOutputFieldsTxt(c_model)
        if os.path.exists(c_file):
            df = pd.read_csv(c_file)
            rms_mean = df["rmse"]
            year_day = [int(x.split("_")[2].split(".")[0])for x in df["File"]]
            plt.figure(figsize=(10,5))
            plt.scatter(year_day, rms_mean)
            plt.axvline(x=year_day[65], c='g')
            plt.axvline(x=year_day[65+9], c='r')
            plt.title(F"RMSE Training, Validation, Test: ({rms_mean.mean():0.3f}, {rms_mean.iloc[65:65+9].mean():0.3f}, {rms_mean.iloc[74:74+9].mean():0.3f}) \n {network} predicted {output_field}")
            plt.savefig(join(imgs_prediction_folder, c_model, "RMSE.png") ,bbox_inches='tight')
            plt.show()
    except Exception as e:
        print(F"Failed for {c_file} {e}")