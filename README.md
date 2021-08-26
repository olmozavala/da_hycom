Folders
===========

Software: `/home/olmozavala/Dropbox/MyProjects/EOAS/COAPS/MURI_AI_Ocean/Data_Assimilation/HYCOM-TSIS`
Outputs: `/data/HYCOM/DA_HYCOM_TSIS`
Summary Results:  https://docs.google.com/spreadsheets/d/1EqkI2K6BA5jnXTny7w_U28vN1JUPn_h0J16SLqV1mgA/edit?usp=sharing

Files
===========

### *0_ComputeMinMaxFromData*
This file can read preprocessed or raw data and obtain the min, max
and variance values for each of the fields specified in `PreprocConfig.py`.
The outputs are saved in a `stats_obs.csv` and `stats_background.csv` file. 
Important!!! you need to add **model** in the first line of both files in order to work. 
Important!!! This file already divides `thknss` by 9806 so we need to be shure this also happens when we read the data.

### *1_AnalizeData*
This file is used to generate multiple plots from the raw data.
Important: this file plots together the fields defined in `PreprocConfig` parameters:
````
PreprocParams.fields_names:     ['u-vel.', 'v-vel.','temp', 'salin', 'thknss', 'srfhgt', 'montg1', 'surflx', 'salflx', 'bl_dpth', 'mix_dpth', 'u_btrop', 'v_btrop'],
PreprocParams.fields_names_obs: ['ssh', 'ssh_err', 'sst', 'sst_err', 'sss', 'sss_err', 'uc', 'vc'],
````

### *2_PreprocData*
This file was used (not anymore) to preprocess the hycom and
tsis outputs into cropped netcdf files. It could reduce
the number of layers and fields. 

### *3_Train_2D and 3D*
This is the main file that trains new models depending on the
`MainConfig_2D.py` configuration file

### *4_TestModel*
**Currently doesn't work!!!!!!**. It reads preproc netcdf files.
This file is used for two options: 
1) Single model evaluation. In this case the user needs to specify wich
weights file to use in `MainConfig.py` and the configuration inside
   this file should match the model being tested. 
   
### *4_TestModel_Whole*
It uses the `get_preproc_config` from `PreprocConfig.py` to identify
which folders to use for reading hycom, tsis, and observation data.
This file is used for two options:
1) Single model evaluation. In this case the user needs to specify wich
   weights file to use in `MainConfig.py` and the configuration inside
   this file should match the model being tested.

### *5_Summary_of_Models*
Iterates over all the trained configurations in the *Training* folder,
and for each configuration it saves the model with the lowest
validation error into a `csv` file. This file is useful because it
contains a list to the best model for each configuration tested. 

### *6_RMSE_Plot*
This file simply plots the RMSE obtained from the
`4_TestModel_Whole.py` file for all the validation examples. 
   
Data
===========

Start date Jan 1st 2009.

Assimilation every 4 days. 

Assimilated variables: T, layer thickness and density 

Hycom: `/Net/gleam/abozec/HYCOM/TSIS/IASx0.03/hindcast_newtsis/gofs30_withpies/archv*`

TSIS: `/Net/gleam/abozec/HYCOM/TSIS/IASx0.03/hindcast_newtsis/gofs30_withpies/incup/incupd*`

Observations (ssh, sst, and T and S profiles): `/data/COAPS_nexsan/people/abozec/TSIS/IASx0.03/obs/qcobs_mdt_gofs/WITH_PIES/tsis_obs*.nc`

**Variables assimialated with TSIS*** are: Temperature (temp), density (calculated from T and S), 
thickness (thknss), and u and v baroclinic (3D). 

Example: 
For the analysis of day 6 the inputs are:
* Model state of day 5  (archv.2009_005)
* Observations of day 5 tsis_obs_ias_2009_0105

TSIS generates file for day 5: incupd.2009_005_00

Available variables in observation's files: **ssh, sst, uc, vc, av_ssh (aviso ssh)**. Also all of them have
an error variable associated with them. 
