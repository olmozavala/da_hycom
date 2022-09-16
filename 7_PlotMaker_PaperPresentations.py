import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr
from hycom.io import read_hycom_fields
import pandas as pd
from viz_utils.eoa_viz import EOAImageVisualizer
import cmocean

## For RMSE summary
rmse_file = "/data/HYCOM/DA_HYCOM_TSIS/PredictionHOY/imgs/0002_GoM2D_STDNORM_PERCOCEAN_0_NET_2DUNET_srfhgt_ssh-ssh-err-sst-sst-err_No-STD_OUT_SRFHGT_384x520/Global_RMSE_and_times.csv"
# df_all = pd.read_csv(rmse_file, parse_dates=[1])
df_all = pd.read_csv(rmse_file)
for year in ("2002", "2006"):
    df = df_all[df_all.File.str.contains(year)]
    fig, ax = plt.subplots(1,1, figsize=(7,5))
    ax.scatter([x.replace("_", "/") for x in df.File], df.rmse, s=1)
    ax.set_xticks(np.round(range(len(df.File))))
    plt.xticks(rotation=30)
    plt.locator_params(nbins=10)
    plt.subplots_adjust(bottom=0.18)
    ax.set_title(F"SSH RMSE for year {year} (mean RMSE: {df.rmse.mean():0.4f} m)")
    ax.set_xlabel("Date")
    ax.set_ylabel("RMSE of SSH in m")
    ax.set_ylim((0.002, 0.010))
    plt.show()

##
df.File


##  For raster examples

# Obs loc /data/COAPS_nexsan/people/abozec/TSIS/GOMb0.04/obs/qcobs_roif/tsis_obs_gomb4_2001070200.nc
# hycom loc /data/COAPS_nexsan/people/abozec/TSIS/GOMb0.04/expt_02.2/incup/incupd.2009_001_18 [a and b]
# tsis loc /data/COAPS_nexsan/people/abozec/TSIS/GOMb0.04/expt_02.2/data/022_archv.2009_001_18 [a and b]
# Available fields on model:     ['montg1', 'srfhgt', 'surflx', 'salflx', 'bl_dpth', 'mix_dpth', 'u_btrop', 'v_btrop', 'u-vel.', 'v-vel.', 'thknss', 'temp', 'salin', 'oneta', 'wtrflx']
# Available fields on observations:   ['ssh', 'ssh_err', 'sst', 'sst_err', 'sss', 'sss_err', 'uc', 'vc', 'av_ssh', 'av_ssh_err', 'val', 'err', 'grdi', 'grdj', 'id', 'ob_grp_present', ]
# Grid /data/COAPS_nexsan/people/abozec/TSIS/GOMb0.04/topo

##
model_file = "/data/COAPS_nexsan/people/abozec/TSIS/GOMb0.04/expt_02.2/incup/incupd.2009_001_18"
tsis_file = "/data/COAPS_nexsan/people/abozec/TSIS/GOMb0.04/expt_02.2/data/022_archv.2009_001_18"
obs_file = "/data/COAPS_nexsan/people/abozec/TSIS/GOMb0.04/obs/qcobs_roif/tsis_obs_gomb4_2009010100.nc"

ssh_model = read_hycom_fields(model_file, ['srfhgt'], [0])
ssh_tsis = read_hycom_fields(tsis_file, ['srfhgt'], [0])
ssh_obs = xr.load_dataset(obs_file)
grid = xr.load_dataset("/data/COAPS_nexsan/people/abozec/TSIS/GOMb0.04/topo/gridinfo.nc")

lats = grid.mplat[:,0]
lons = grid.mplon[0,:]

## SSH
img = plt.imread('/home/olmozavala/Dropbox/TutorialsByMe/Python/PythonExamples/Python/MatplotlibEx/map_backgrounds/bluemarble.png')
extent = (lons.min(), lons.max(), lats.min(), lats.max())
img_extent = (-180, 180, -90, 90)
fig, ax = plt.subplots(1, 1, figsize=(10,6), subplot_kw={'projection': ccrs.PlateCarree()})
ax.stock_img()
ax.imshow(img, origin='upper', extent=img_extent, transform=ccrs.PlateCarree())
levels = np.around(np.linspace(-.6,.6,80), 2)
im = ax.contourf(lons, lats, ssh_model['srfhgt'][0,:,:], levels=levels, cmap=cmocean.cm.delta, extent=extent)
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