from os.path import join
from constants_proj.AI_proj_params import *
from img_viz.constants import PlotMode
import numpy as np

# ----------------------------- UM -----------------------------------
_output_folder = '/data/HYCOM/DA_HYCOM_TSIS/'  # Where to save everything

def get_preproc_config():
    model_config = {
        # PreprocParams.input_folder_hycom: '/data/COAPS_Net/gleam/abozec/HYCOM/TSIS/IASx0.03/hindcast_newtsis/gofs30_withpies',
        # PreprocParams.input_folder_hycom: '/data/COAPS_nexsan/people/abozec/TSIS/GOMb0.04/expt_01.0/data',
        PreprocParams.input_folder_hycom: '/data/COAPS_nexsan/people/abozec/TSIS/GOMb0.04/expt_02.0/data',
        # PreprocParams.input_folder_tsis: '/data/COAPS_Net/gleam/abozec/HYCOM/TSIS/IASx0.03/hindcast_newtsis/gofs30_withpies/incup',
        # PreprocParams.input_folder_tsis: '/data/COAPS_nexsan/people/abozec/TSIS/GOMb0.04/expt_01.0/incup',
        PreprocParams.input_folder_tsis: '/data/COAPS_nexsan/people/abozec/TSIS/GOMb0.04/expt_02.0/incup',
        # PreprocParams.input_folder_obs: '/data/COAPS_nexsan/people/abozec/TSIS/IASx0.03/obs/qcobs_mdt_gofs/WITH_PIES',
        PreprocParams.input_folder_obs: '/data/COAPS_nexsan/people/abozec/TSIS/GOMb0.04/obs/qcobs_roif',
        PreprocParams.output_folder: join(_output_folder, 'preproctest'),
        PreprocParams.imgs_output_folder: join(_output_folder, 'preproc', 'imgs'),
        PreprocParams.YEARS: [2009],
        # PreprocParams.MONTHS: range(1, 13),
        PreprocParams.MONTHS: [1],

        # Available fields on increment: ['montg1', 'srfhgt', 'surflx', 'salflx', 'bl_dpth', 'mix_dpth', 'u_btrop', 'v_btrop', 'u-vel.', 'v-vel.', 'thknss', 'temp', 'salin']
        # Available fields on model:     ['montg1', 'srfhgt', 'surflx', 'salflx', 'bl_dpth', 'mix_dpth', 'u_btrop', 'v_btrop', 'u-vel.', 'v-vel.', 'thknss', 'temp', 'salin', 'oneta', 'wtrflx']
        # Available fields on observations:   ['ssh', 'ssh_err', 'sst', 'sst_err', 'sss', 'sss_err', 'uc', 'vc', 'av_ssh', 'av_ssh_err', 'val', 'err', 'grdi', 'grdj', 'id', 'ob_grp_present', ]

        # There MUST be the same number of field names between the two because there is a comparison between them.
        # PreprocParams.fields_names:     ['u-vel.', 'v-vel.','temp', 'salin', 'thknss', 'srfhgt', 'montg1', 'surflx', 'salflx', 'bl_dpth', 'mix_dpth', 'u_btrop', 'v_btrop'],
        # PreprocParams.fields_names_obs: ['ssh', 'ssh_err', 'sst', 'sst_err', 'sss', 'sss_err', 'uc', 'vc'],
        # PreprocParams.fields_names:     ['thknss', 'srfhgt', 'temp', 'u-vel.', 'v-vel.','salin', 'mix_dpth'],
        PreprocParams.fields_names:     ['temp'],
        PreprocParams.fields_names_obs: ['ssh', 'sst'],
        # PreprocParams.plot_modes_per_field: [PlotMode.RASTER, PlotMode.MERGED, PlotMode.RASTER, PlotMode.RASTER, PlotMode.RASTER],
        PreprocParams.plot_modes_per_field: [PlotMode.RASTER for x in range(14)],
        # PreprocParams.layers_to_plot: [0], # Total of 41
        PreprocParams.layers_to_plot: np.arange(0,5,1), # Total of 41
        # layer 0 --> z-coordinate everywhere  (1 mt depth)
        # layer 1 to 16 --> sigma- coordinate for 'shallow areas' and z coordinates inside the Gulf (deeper)
        # layer 16 to 41 --> ro- coordinates everywhere.
        # 385, 525, 41

 # 'montg1'  --> ?
 # 'srfhgt'
 # 'surflx'  --> Flujos de calor superficie: calor sensible o calor latente (tal vez suma)
 # 'salflx'  --> Flujos de salinidad superficie: evaporacion menos precipitacion
 # 'bl_dpth' --> Profundidad de la capa limite
 # 'mix_dpth'
 # 'u_btrop' --> Promedio de toda la vertical
 # 'v_btrop'
 # 'u-vel.'  -->
 # 'v-vel.'
 # 'thknss'
 # 'temp'
 # 'salin'
    }
    return model_config

