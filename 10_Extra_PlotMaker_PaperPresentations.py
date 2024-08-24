# %%
import os
import sys
sys.path.append("eoas_pyutils/")
sys.path.append("eoas_pyutils/hycom_utils/python")
from os.path import join

from hycom.io import read_hycom_fields, subset_hycom_field, read_hycom_coords
from hycom.info import read_field_names
import pandas as pd
import xarray as xr
import numpy as np
from viz_utils.eoa_viz import EOAImageVisualizer
from viz_utils.constants import PlotMode, BackgroundType
from proc_utils.gom import lc_from_ssh
from shapely.geometry import LineString, Polygon
import cmocean.cm as cm
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# %% ================== Plot Gom Domain Oringal==============
bbox = [-98, -77.04, 18.091648, 31.960648]
def gomdomain():
    output_folder = "/home/olmozavala/Dropbox/MyProjects/EOAS/COAPS/MURI_AI_Ocean/Data_Assimilation/HYCOM-TSIS/ImagesForPresentation/"
    model_file = "/data/COAPS_nexsan/people/abozec/TSIS/GOMb0.04/expt_02.2/incup/incupd.2010_003_18"
    hycom_fields = read_hycom_fields(model_file, ['temp','srfhgt'], [0])
    grid = xr.load_dataset("/data/COAPS_nexsan/people/abozec/TSIS/GOMb0.04/topo/gridinfo.nc")
    hycom_fields['temp'][np.isnan(hycom_fields['srfhgt'])] = np.nan

    lats = grid.mplat[:,0]
    lons = grid.mplon[0,:]

    # Obtain BBOX from lats and lons
    print(f"bbox: {bbox}")
    viz_obj = EOAImageVisualizer(lats=lats, lons=lons, disp_images=True, output_folder=output_folder,
                                 units='$^\circ$C',
                                 background=BackgroundType.WHITE,
                                 contourf=True,
                                 coastline=True)
    viz_obj.plot_3d_data_npdict(hycom_fields, ['temp'], [0], 'Gulf of Mexico domain', 'f01',
                                plot_mode=PlotMode.MERGED,
                                mincbar=[15], maxcbar=[27])
    
gomdomain()

# %% ================== Plot Gom Domain Bathymetry==============
bat_file = "/home/olmozavala/Dropbox/TestData/netCDF/Bathymetry/w100n40.nc"
bat_data = xr.load_dataset(bat_file)
# Crop to bbox
bat_data = bat_data.sel(x=slice(bbox[0], bbox[1]), y=slice(bbox[2], bbox[3]))
# %% Plot with cartopy 
fig = plt.figure(figsize=(12, 10))
ax = plt.axes(projection=ccrs.PlateCarree())

# Assuming bbox contains the extents [longitude_min, longitude_max, latitude_min, latitude_max]
bbox = [-98, -77.04, 18.09, 31.96]
ax.set_extent(bbox)

# Plot iso-contours of bathymetry data
contour_levels = np.linspace(-5000, 0, 100)  # Adjust range and number of levels as needed
im = ax.contourf(bat_data.x, bat_data.y, bat_data.z, levels=contour_levels, cmap=cm.deep, alpha=0.4)

# Plot contour lines above the filled contour
contour_levels = [-5000, -2000, -1000, -500, -200, -100, -5, 0]
cont_lines = ax.contour(bat_data.x, bat_data.y, bat_data.z, levels=contour_levels, colors='k', linewidths=0.5)
ax.clabel(cont_lines, inline=True, fontsize=10, fmt='%1.0f m', colors='white')  # Label the contours

# Set title and coastlines
ax.set_title("Gulf of Mexico Domain", fontsize=24)
ax.coastlines(resolution='10m', color='black', linewidth=1)

# Customize gridlines
gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False  # Disable labels at the top
gl.right_labels = False  # Disable labels on the right
gl.xlabel_style = {'size': 18, 'color': 'black'}  # Customize font size and color of x labels
gl.ylabel_style = {'size': 18, 'color': 'black'}  # Customize font size and color of y labels
plt.tight_layout()

# Save the plot without extra white space
plt.savefig('/home/olmozavala/Dropbox/Apps/Overleaf/CNN_DA/images/f01.png', bbox_inches='tight', pad_inches=.1)

plt.show()
# %% ================= Similar plot but with random boxes of size 384x520

for desired_size in [0, 160, 120, 80]:
    fig = plt.figure(figsize=(12, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Assuming bbox contains the extents [longitude_min, longitude_max, latitude_min, latitude_max]
    bbox = [-98, -77.04, 18.09, 31.96]
    ax.set_extent(bbox)

    model_file = "/data/COAPS_nexsan/people/abozec/TSIS/GOMb0.04/expt_02.2/incup/incupd.2010_003_18"
    hycom_fields = read_hycom_fields(model_file, ['temp','srfhgt'], [0])
    grid = xr.load_dataset("/data/COAPS_nexsan/people/abozec/TSIS/GOMb0.04/topo/gridinfo.nc")
    hycom_fields['temp'][np.isnan(hycom_fields['srfhgt'])] = np.nan

    # Plot temperature using contourf
    contour_levels = np.linspace(15, 28, 100)  # Adjust range and number of levels as needed
    im = ax.contourf(grid.mplon, grid.mplat, hycom_fields['temp'][0,:,:], levels=contour_levels, cmap=cm.thermal, alpha=0.9)

    # Plot contour lines above the filled contour
    contour_levels = [-5000, -2000, -1000, -500, -200, -100, -5, 0]
    cont_lines = ax.contour(bat_data.x, bat_data.y, bat_data.z, levels=contour_levels, colors='k', linewidths=0.5, alpha=0.8)
    ax.clabel(cont_lines, inline=True, fontsize=8, fmt='%1.0f m', colors='white')  # Label the contours

    long_min, long_max, lat_min, lat_max = [bbox[0], bbox[1], bbox[2], bbox[3]]

    # Generate random boxes of size 120x120 pixels. We need to tranform the pixel size to degrees
    # Since the size of the figure is 385x520 pixels, we need to divide the size of the domain by the size of the figure
    # to get the size of the box in degrees
    line_width = 5.5
    title_font = 50
    xsize = (desired_size/520)*(long_max - long_min)
    ysize = (desired_size/385)*(lat_max - lat_min)
    for i in range(6):
        # Generate random longitude and latitude within the defined bounds
        lon0 = np.random.uniform(long_min, long_max - xsize)  # Subtract to avoid going out of bounds
        lat0 = np.random.uniform(lat_min, lat_max - ysize)
        lon1 = lon0 + xsize  # Width of 1 degree longitude
        lat1 = lat0 + ysize  # Height of 1 degree latitude

        # Plot box
        ax.add_patch(plt.Rectangle((lon0, lat0), lon1-lon0, lat1-lat0, edgecolor='red', facecolor='none', linewidth=line_width, transform=ccrs.PlateCarree()))

    # Add patch for the full domain

    # Set title and coastlines
    if desired_size == 0:
        border = .08
        ax.add_patch(plt.Rectangle((long_min+border, lat_min+border), long_max-long_min-2*border, lat_max-lat_min-2*border, 
                                edgecolor='red', facecolor='none', linewidth=line_width, transform=ccrs.PlateCarree()))
        ax.set_title(f"Window size 385x520", fontsize=title_font)
    else:
        ax.set_title(f"Window size {desired_size}x{desired_size}", fontsize=title_font)
    ax.coastlines(resolution='10m', color='black', linewidth=1)

    # Customize gridlines
    gl = ax.gridlines(draw_labels=False, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False  # Disable labels at the top
    gl.right_labels = False  # Disable labels on the right
    gl.xlabel_style = {'size': 14, 'color': 'black'}  # Customize font size and color of x labels
    gl.ylabel_style = {'size': 14, 'color': 'black'}  # Customize font size and color of y labels
    plt.tight_layout()

    # Save the plot without extra white space
    plt.savefig(f'/home/olmozavala/Dropbox/Apps/Overleaf/CNN_DA/images/f03_{desired_size}.png', bbox_inches='tight', pad_inches=.1)

    plt.show()

# %% Merge images vertically from the terminal
root_folder = '/home/olmozavala/Dropbox/Apps/Overleaf/CNN_DA/images'
cmd = f"convert "
for desired_size in [0, 160, 120, 80]:
    cmd += f"{root_folder}/f03_{desired_size}.png {root_folder}/spacer.png "
cmd += f"+append {root_folder}/f03.png"
print(cmd)
# Runc command
os.system(cmd)

# %% ================= In case you want to add one example of how the LC is in extended or retracted state
def LC_plot(dates, dates_str):
    output_folder = "/home/olmozavala/Dropbox/Apps/Overleaf/Book_Chapter_CNN_DA/imgs"
    model_folder = "/data/COAPS_nexsan/people/abozec/TSIS/GOMb0.04/expt_02.2/incup/"

    all_files = os.listdir(model_folder)
    all_files.sort()
    for i, c_date in enumerate(dates):
        model_file = f'incupd.{c_date}_18'
        hycom_fields = read_hycom_fields(join(model_folder, model_file), ['srfhgt'], [0])
        ssh = hycom_fields['srfhgt'][0,:,:]

        if i == 0:
            grid = xr.load_dataset("/data/COAPS_nexsan/people/abozec/TSIS/GOMb0.04/topo/gridinfo.nc")
            lats = grid.mplat[:,0]
            lons = grid.mplon[0,:]

        viz_obj = EOAImageVisualizer(lats=lats, lons=lons, disp_images=True, output_folder=output_folder,
                                        units='m',
                                        background=BackgroundType.WHITE,
                                        contourf=True,
                                        coastline=True)

        lc = lc_from_ssh(ssh, lons, lats, np.nanmean(ssh))

        mylinestring = LineString(list(lc))
        viz_obj.__setattr__('additional_polygons', [mylinestring])
        viz_obj.plot_3d_data_npdict(hycom_fields, ['srfhgt'], [0], title=f'Gulf of Mexico SSH {dates_str[i]}',
                                    file_name_prefix=f'GoM_Domain_{dates_str[i]}', mincbar=[15], maxcbar=[27])


dates = ['2002_110', '2006_110']
dates_str = ['2002-04-20', '2006-04-20']
LC_plot(dates, dates_str)

# %% ================  For RMSE summary
def RMSE_Plot(file_name):
    rmse_file = "/data/HYCOM/DA_HYCOM_TSIS/PredictionBestPaper/imgs/0001_GoM2D_STDNORM_PERCOCEAN_0_NET_2DUNET_srfhgt_ssh-ssh-err-sst-sst-err_No-STD_OUT_SRFHGT_384x520/Global_RMSE_and_times_2010_BK.csv"
    # rmse_file = "/data/HYCOM/DA_HYCOM_TSIS/Prediction2002_2006/imgs/0001_GoM2D_STDNORM_PERCOCEAN_0_NET_2DUNET_srfhgt_ssh-ssh-err-sst-sst-err_No-STD_OUT_SRFHGT_384x520/Global_RMSE_and_times_2002_2006_BK.csv"
    df_all = pd.read_csv(rmse_file)
    # for year in ["2002", "2006"]:
    for year in ["2010"]:
        df = df_all[df_all.File.str.contains(year)]
        # Remove outliers
        df = df[df.rmse < 0.006]

        x = [x.replace("_", "/") for x in df.dates]
        y = df.rmse

        fig, ax = plt.subplots(1,1, figsize=(9,5))
        ax.scatter(x, y*1000,  s=2)
        # Make a vertical line after the first 292 elements
        ax.plot([x[292], x[292]], [0, 10], color='green', linestyle='-', linewidth=1)
        mean_train = np.mean(y[0:292])*1000
        mean_val = np.mean(y[292:])*1000
        ax.plot([x[0], x[-1]], [mean_train, mean_train], color='red', linestyle='--', linewidth=1,
                label=F"Mean Training RMSE: {mean_train:0.2f} mm")
        ax.plot([x[0], x[-1]], [mean_val, mean_val], color='darkseagreen', linestyle='--', linewidth=1,
                label=F"Mean Test RMSE: {mean_val:0.2f} mm")
        # mean_all = np.mean(y)*1000
        # ax.plot([x[0], x[-1]], [mean_all, mean_all], color='red', linestyle='--', linewidth=1,
                # label=F"Mean validation: {mean_all:0.2f} mm")
        ax.set_xticks(np.round(range(len(df.dates))))
        plt.xticks(rotation=25)
        plt.locator_params(nbins=15)
        plt.subplots_adjust(bottom=0.18)
        # ax.set_title(F"SSH RMSE for year {year} (mean RMSE: {df.rmse.mean()*1000:0.2f} mm)")
        # ax.set_title(F"Sea Surface Hight RMSE for year {year} training and validation sets")
        ax.set_title(F"Sea Surface Hight RMSE for year {year}", fontsize=16)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("RMSE of SSH in mm", fontsize=12)
        ax.set_ylim((2.8, 6))
        # Add legend
        ax.legend(loc='upper left', fontsize=10)
        # Tight plot
        plt.tight_layout()
        # Save figure
        plt.savefig(f'{file_name}_{year}.png', dpi=300)
        plt.show()

file_name = "/home/olmozavala/Dropbox/Apps/Overleaf/Book_Chapter_CNN_DA/imgs/GoM_RMSE"
RMSE_Plot(file_name)

# %%
print(x)


#%%  For raster examples
# Obs loc /data/COAPS_nexsan/people/abozec/TSIS/GOMb0.04/obs/qcobs_roif/tsis_obs_gomb4_2001070200.nc
# hycom loc /data/COAPS_nexsan/people/abozec/TSIS/GOMb0.04/expt_02.2/incup/incupd.2009_001_18 [a and b]
# tsis loc /data/COAPS_nexsan/people/abozec/TSIS/GOMb0.04/expt_02.2/data/022_archv.2009_001_18 [a and b]
# Available fields on model:     ['montg1', 'srfhgt', 'surflx', 'salflx', 'bl_dpth', 'mix_dpth', 'u_btrop', 'v_btrop', 'u-vel.', 'v-vel.', 'thknss', 'temp', 'salin', 'oneta', 'wtrflx']
# Available fields on observations:   ['ssh', 'ssh_err', 'sst', 'sst_err', 'sss', 'sss_err', 'uc', 'vc', 'av_ssh', 'av_ssh_err', 'val', 'err', 'grdi', 'grdj', 'id', 'ob_grp_present', ]
# Grid /data/COAPS_nexsan/people/abozec/TSIS/GOMb0.04/topo

#%%
model_file = "/data/COAPS_nexsan/people/abozec/TSIS/GOMb0.04/expt_02.2/incup/incupd.2009_001_18"
# tsis_file = "/data/COAPS_nexsan/people/abozec/TSIS/GOMb0.04/expt_02.2/data/022_archv.2009_001_18"
tsis_file = "/data/COAPS_nexsan/archive/dvoss/TSIS/GOMb0.04/expt_02.2/incup/incupd.2006_119_18"
obs_file = "/data/COAPS_nexsan/people/abozec/TSIS/GOMb0.04/obs/qcobs_roif/tsis_obs_gomb4_2009010100.nc"

# ssh_model = read_hycom_fields(model_file, ['srfhgt'], [0])
hycom_fields = read_hycom_fields(tsis_file, ['srfhgt', 'temp'], [0])
ssh_obs = xr.load_dataset(obs_file)
grid = xr.load_dataset("/data/COAPS_nexsan/people/abozec/TSIS/GOMb0.04/topo/gridinfo.nc")

lats = grid.mplat[:,0]
lons = grid.mplon[0,:]

#%% SSH
img = plt.imread('/home/olmozavala/Dropbox/TutorialsByMe/Python/PythonExamples/Python/MatplotlibEx/map_backgrounds/bluemarble.png')
extent = (lons.min(), lons.max(), lats.min(), lats.max())
img_extent = (-180, 180, -90, 90)
# im = ax.contourf(lons, lats, ssh_model['srfhgt'][0,:,:], levels=levels, cmap=cm.delta, extent=extent)
# .stock_img()extent, transform=ccrs.PlateCarsrfhg), cmap=cm.delta, extent=extent)
im = ax.imshow(hycom_fields['temp'][0,:,:]
levels = np.around(np.linspace(-.6,.6,80), 2)
# im = ax.contourf(lons, lats, ssh_model['srfhgt'][0,:,:], levels=levels, cmap=cm.delta, extent=extent)
# im = ax.contourf(lons, lats, hycom_fields['srfhgt'][0,:,:], levels=levels, cmap=cm.delta, extent=extent)
im = ax.imshow(hycom_fields['temp'][0,:,:], cmap=cm.thermal, extent=extent)
ax.set_title("Sea Surface Height")
gl = ax.gridlines(draw_labels=True, color='grey', alpha=0.3, linestyle='--')
gl.top_labels = False
gl.left_labels = False
plt.colorbar(im, location='right', shrink=.7, pad=.12, label='Meters')
plt.tight_layout()
plt.savefig('/home/olmozavala/Dropbox/MyConferencesAndWorkshops/2022/OceanSciences/CNN_DA/imgs/ssh.png')
plt.show()
plt.close()
print("Done!")