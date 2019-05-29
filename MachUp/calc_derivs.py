import numpy as np
import json
import os
import copy

def run_perturbed_position(configs,pos,perturbation):
    perturb_forward = copy.deepcopy(configs)
    perturb_backward = copy.deepcopy(configs)

    for key in configs["wings"]:
        if "tail" in key:
            print(key)
            perturb_forward["wings"][key]["connect"][pos] = base_configs["wings"][key]["connect"][pos]+perturbation
            perturb_backward["wings"][key]["connect"][pos] = base_configs["wings"][key]["connect"][pos]-perturbation

    forward_file = pos+"forward.json"
    with open(forward_file, 'w', newline='\r\n') as dump_file:
        json.dump(perturb_forward,dump_file,indent=4)

    backward_file = pos+"backward.json"
    with open(backward_file, 'w', newline='\r\n') as dump_file:
        json.dump(perturb_backward,dump_file,indent=4)

    os.system("./Machup.out "+forward_file)
    os.system("./Machup.out "+backward_file)

    return forward_file, backward_file

def run_base_case(configs):

    configs["run"] = {"forces":""}
    configs["solver"]["convergence"] = 1e-4

    base_file = "base.json"
    with open(base_file,'w') as dump_file:
        json.dump(configs,dump_file,indent=4)

    os.system("./Machup.out "+base_file)

def get_forces_and_moments(filename,configs):
    filename = filename.replace(".json","_forces.json")

    with open(filename, 'r') as coefs_file:
        coefs = json.load(coefs_file)
    
    q_inf = 0.5*configs["condition"]["density"]*configs["condition"]["V_ref"]
    S_w = configs["reference"]["area"]
    l_ref_lon = configs["reference"]["longitudinal_length"]
    l_ref_lat = configs["reference"]["lateral_length"]

    lead_FM = np.zeros((2,3))
    tail_FM = np.zeros((2,3))

    for key in coefs["total"]:
        if "lead" in key and "tail" not in key:
            lead_FM[0,0] += coefs["total"][key]["CX"]*q_inf*S_w
            lead_FM[0,1] += coefs["total"][key]["CY"]*q_inf*S_w
            lead_FM[0,2] += coefs["total"][key]["CZ"]*q_inf*S_w
            lead_FM[1,0] += coefs["total"][key]["Cl"]*q_inf*S_w*l_ref_lat
            lead_FM[1,1] += coefs["total"][key]["Cm"]*q_inf*S_w*l_ref_lon
            lead_FM[1,2] += coefs["total"][key]["Cn"]*q_inf*S_w*l_ref_lat
        if "tail" in key and "lead" not in key:
            tail_FM[0,0] += coefs["total"][key]["CX"]*q_inf*S_w
            tail_FM[0,1] += coefs["total"][key]["CY"]*q_inf*S_w
            tail_FM[0,2] += coefs["total"][key]["CZ"]*q_inf*S_w
            tail_FM[1,0] += coefs["total"][key]["Cl"]*q_inf*S_w*l_ref_lat
            tail_FM[1,1] += coefs["total"][key]["Cm"]*q_inf*S_w*l_ref_lon
            tail_FM[1,2] += coefs["total"][key]["Cn"]*q_inf*S_w*l_ref_lat

    return lead_FM,tail_FM

def calc_cent_diff(configs,wrt,incr):
    if wrt in "dxdydz":
        temp = run_perturbed_position(base_configs,wrt,incr)
    else:
        raise ValueError("Cannot take a derivative with respect to the spcified variable.")
    forward_file = temp[0]
    backward_file = temp[1]
    fwd_lead,fwd_tail = get_forces_and_moments(forward_file,configs)
    bwd_lead,bwd_tail = get_forces_and_moments(backward_file,configs)

    derivs = (fwd_tail-bwd_tail)/(2*incr)
    return derivs

if __name__=="__main__":

    # Find the matrix of derivatives of aerodynamic forces and moments with respect to position and orientation.

    # Initialize
    os.system("rm *_forces.json")
    base_config_file = 'input.json'
    with open(base_config_file) as base_file:
        base_configs = json.load(base_file)

    run_base_case(base_configs)

    # Specify increments
    dx = 0.001
    dtheta = 0.001

    # d/dx
    dFM_dx = calc_cent_diff(base_configs,"dx",dx)

    # d/dy
    dFM_dy = calc_cent_diff(base_configs,"dy",dx)

    # d/dz
    dFM_dz = calc_cent_diff(base_configs,"dz",dx)

    print("Derivatives w.r.t. x:\n{0}".format(dFM_dx))
    print("Derivatives w.r.t. y:\n{0}".format(dFM_dy))
    print("Derivatives w.r.t. z:\n{0}".format(dFM_dz))
    
    F_pbar = np.asarray([dFM_dx.flatten(),dFM_dy.flatten(),dFM_dz.flatten()])
    print("Fp:\n{0}".format(F_pbar))