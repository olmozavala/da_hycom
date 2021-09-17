from constants.AI_params import AiModels
from constants_proj.AI_proj_params import NetworkTypes

def getCompleteNameTxt(name):
    _run_name = F'RUN_NETWORK_IN_No-STD_OUTPUT_160x160'
    sections = name.split('_')
    id = sections[0]
    netname = getNetworkTypeTxt(name)
    in_names = getInputFieldsTxt(name)
    out_names = getOutputFieldsTxt(name)

    final_name = _run_name.replace("RUN", id)
    final_name = final_name.replace("NETWORK", netname)
    final_name = final_name.replace("IN", F"IN_{in_names}")
    final_name = final_name.replace("OUTPUT", F"OUT_{out_names}")
    return final_name

def getNetworkTypeTxt(name):
    sections = name[name.find("NET"):].split("_")
    if "simplecnn" in name.lower():
        return F"{sections[1]}_{sections[2]}"
    else:
        return F"{sections[1]}"
    return "Unknown"

def getNeworkArchitectureAndTypeFromName(networkName):
    if "Simple" in networkName:
        if "2" in networkName:
            return "SimpleCNN", NetworkTypes.SimpleCNN_2
        if "4" in networkName:
            return "SimpleCNN", NetworkTypes.SimpleCNN_4
        if "8" in networkName:
            return "SimpleCNN", NetworkTypes.SimpleCNN_8
        if "16" in networkName:
            return "SimpleCNN", NetworkTypes.SimpleCNN_16
    else:
        return AiModels.UNET_2D_SINGLE, NetworkTypes.UNET

def getInputFieldsTxt(name):
    sections = name[name.find("IN"):].split("_")
    in_fields = sections[1]
    if "WSSH" in name:
        in_fields += ", SSH"
    if "LATLON" in name:
        in_fields += ", LATLON"
    if ("IN2" in name) or ("IN8" in name):
        in_fields += "-difssh"
    return in_fields

def getObsFieldsTxt(name):
    sections = name[name.find("IN"):].split("_")
    all_inputs = sections[1]
    if all_inputs.find("-") != -1:
        all_inputs = all_inputs.replace("-err","_err")
        return all_inputs.split("-")
    else:
        return [all_inputs]

def getOutputFields(name):
    # ProjTrainingParams.output_fields: ['temp', 'srfhgt', 'salin', 'u-vel.', 'v-vel.']
    sections = name[name.find("OUT"):].split("_")
    return [sections[1].lower()]

def getOutputFieldsTxt(name):
    sections = name[name.find("OUT"):].split("_")
    return sections[1]

def getId(name):
    sections = name.split("_")
    return sections[0]

def getBBOXandText(name):
    bbox = name.split("_")[-6].split("x")
    return int(bbox[0]), int(bbox[1]), F"{int(bbox[0]):03d}x{int(bbox[1]):03d}"

def getPercOcean(name):
    percocean_orig = name[name.find("PERCOCEAN"):].split("_")[1]
    perocean_number = int(percocean_orig)/10
    perocean_str = str(perocean_number)
    return perocean_number, perocean_str

def landperc(name):
    if "no_land" in name.lower():
        return "No Land"
    return "Unknown"