import matplotlib.pyplot as plt
from inout.io_netcdf import read_netcdf
import numpy as np
import matplotlib.patches as patches

from os.path import join
import os


input_folder = "/home/olmozavala/Dropbox/MyProjects/EOAS/COAPS/MURI_AI_Ocean/Data_Assimilation/HYCOM-TSIS/testdata"
input_file =  "model_2009_205.nc"

data = read_netcdf(join(input_folder,input_file), ["temp"])

fig, ax = plt.subplots()
ax.imshow(np.flip(data["temp"], axis=0))
print(data["temp"].shape)

found = 0
while found < 20:
    row = np.random.randint(0,891)
    col = np.random.randint(0,1401)
    ldata = data["temp"][row:row+160, col:col+160]
    if len(ldata[ldata.mask]) == 0:
        # Create a Rectangle patch
        print(F"Adding at: row:{row}-{row+160} and col:{col}-{col+160}")
        if row > 445:
            rect = patches.Rectangle((col, int(np.abs(445-row))), 160, 160, linewidth=1, edgecolor='r', facecolor='none')
        else:
            rect = patches.Rectangle((col, int(np.abs(445-row))), 160, -160, linewidth=1, edgecolor='r', facecolor='none')
        # rect = patches.Rectangle((col, row), 160, 160, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        found += 1

# rect = patches.Rectangle((100, 120), 160, 160, linewidth=1, edgecolor='r', facecolor='none')
# ax.add_patch(rect)
# rect = patches.Rectangle((120, 150), 160, 160, linewidth=1, edgecolor='r', facecolor='none')
# ax.add_patch(rect)
# rect = patches.Rectangle((650, 20), 160, 160, linewidth=1, edgecolor='r', facecolor='none')
# ax.add_patch(rect)
# rect = patches.Rectangle((1200, 300), 160, 160, linewidth=1, edgecolor='r', facecolor='none')
# ax.add_patch(rect)
plt.show()