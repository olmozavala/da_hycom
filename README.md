Folders
===========

Software: `/home/olmozavala/Dropbox/MyProjects/EOAS/COAPS/MURI_AI_Ocean/Data_Assimilation/HYCOM-TSIS`
Outputs: `/data/HYCOM/DA_HYCOM_TSIS`
Summary Results:  https://docs.google.com/spreadsheets/d/1EqkI2K6BA5jnXTny7w_U28vN1JUPn_h0J16SLqV1mgA/edit?usp=sharing


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
