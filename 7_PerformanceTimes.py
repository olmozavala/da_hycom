import numpy as np
import os
from os.path import  join
import pandas as pd

root_folder = "/data/HYCOM/DA_HYCOM_TSIS/Prediction/imgs"
all_folders = os.listdir(root_folder)

all_folders = [x for x in all_folders if os.path.isdir(join(root_folder, x))]

for c_folder in all_folders:
    file_name = join(root_folder, c_folder, "Global_RMSE_and_times.csv")
    df = pd.read_csv(file_name)
    mtime = df['times mean'].mean()
    std_time = df['times mean'].std()
    print(F"For {c_folder}: mean: {mtime:0.4f} std: {std_time:0.4f}")

