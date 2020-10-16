from os.path import join
from constants_proj.AI_proj_params import *
from img_viz.constants import PlotMode

# ----------------------------- UM -----------------------------------
_output_folder = '/data/HYCOM/DA_HYCOM_TSIS/'  # Where to save everything

def get_preproc_config():
    model_config = {
        PreprocParams.input_folder_hycom: '/data/COAPS_Net/gleam/abozec/HYCOM/TSIS/IASx0.03/hindcast_newtsis/gofs30_withpies',
        PreprocParams.input_folder_tsis: '/data/COAPS_Net/gleam/abozec/HYCOM/TSIS/IASx0.03/hindcast_newtsis/gofs30_withpies/incup',
        PreprocParams.input_folder_obs: '/data/COAPS_nexsan/people/abozec/TSIS/IASx0.03/obs/qcobs_mdt_gofs/WITH_PIES',
        PreprocParams.output_folder: join(_output_folder, 'preproc'),
        PreprocParams.imgs_output_folder: join(_output_folder, 'preproc', 'imgs'),
        PreprocParams.YEARS: [2009],
        PreprocParams.MONTHS: range(1, 13),
        # There MUST be the same number of field names between the two because there is a comparison between them.
        PreprocParams.fields_names: ['temp', 'srfhgt', 'salin',  'u-vel.', 'v-vel.'], #'thknss',
        PreprocParams.fields_names_obs: ['sst', 'ssh', 'sss', 'uc', 'vc'],
        # PreprocParams.fields_names: ['srfhgt'],
        # PreprocParams.fields_names_obs: ['ssh'],
        PreprocParams.plot_modes_per_field: [PlotMode.RASTER, PlotMode.MERGED, PlotMode.RASTER, PlotMode.RASTER, PlotMode.RASTER],
        PreprocParams.layers_to_plot: [0],
    }
    return model_config

