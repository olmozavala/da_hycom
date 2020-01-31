from os.path import join
from constants_proj.AI_proj_params import *
from img_viz.constants import PlotMode

# ----------------------------- UM -----------------------------------
_output_folder = '/data/HYCOM/DA_HYCOM_TSIS/'  # Where to save everything

def get_paralallel_config():
    model_config = {
        ParallelParams.NUM_PROC: 10
    }
    return model_config

def get_preproc_config():
    model_config = {
        PreprocParams.input_folder_tsis: '/data/COAPS_Net/gleam/dmitry/hycom/TSIS/IASx0.03/output/',
        PreprocParams.input_folder_obs: '/data/COAPS_nexsan/people/abozec/TSIS/IASx0.03/obs/qcobs_mdt_gofs/',
        PreprocParams.output_folder: join(_output_folder, 'preproc'),
        PreprocParams.PIES: [True],
        PreprocParams.YEARS: [2009, 2010, 2011],
        PreprocParams.MONTHS: range(1, 13),
        # There MUST be the same number of field names between the two because there is a comparison betwen them.
        PreprocParams.fields_names: ['temp', 'srfhgt', 'salin', 'u-vel.', 'v-vel.'],
        PreprocParams.fields_names_obs: ['sst', 'ssh', 'sss', 'uc', 'vc'],
        PreprocParams.layers_to_plot: [0],
    }
    return model_config

