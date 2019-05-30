import numpy as np
import json
import os
import copy

def run_perturbed_position(configs,pos,perturbation):
    perturb_forward = copy.deepcopy(configs)
    perturb_backward = copy.deepcopy(configs)

    for key in configs["wings"]:
        if "tail" in key:
            perturb_forward["wings"][key]["connect"][pos] = trim_configs["wings"][key]["connect"][pos]+perturbation
            perturb_backward["wings"][key]["connect"][pos] = trim_configs["wings"][key]["connect"][pos]-perturbation

    forward_file = pos+"_forward.json"
    with open(forward_file, 'w', newline='\r\n') as dump_file:
        json.dump(perturb_forward,dump_file,indent=4)

    backward_file = pos+"_backward.json"
    with open(backward_file, 'w', newline='\r\n') as dump_file:
        json.dump(perturb_backward,dump_file,indent=4)

    os.system("./Machup.out "+forward_file)
    os.system("./Machup.out "+backward_file)

    return forward_file, backward_file

def run_perturbed_orientation(configs,ori,perturbation):
    perturb_forward = copy.deepcopy(configs)
    perturb_backward = copy.deepcopy(configs)

    for key in configs["wings"]:
        if "tail" in key:
            print(key)

def run_perturbed_control(configs,control,perturbation):
    perturb_forward = copy.deepcopy(configs)
    perturb_backward = copy.deepcopy(configs)

    control_key = "t_"+control

    perturb_forward["controls"][control_key]["deflection"] = trim_configs["controls"][control_key]["deflection"]+perturbation
    perturb_backward["controls"][control_key]["deflection"] = trim_configs["controls"][control_key]["deflection"]-perturbation

    forward_file = control+"_forward.json"
    with open(forward_file, 'w', newline='\r\n') as dump_file:
        json.dump(perturb_forward,dump_file,indent=4)

    backward_file = control+"_backward.json"
    with open(backward_file, 'w', newline='\r\n') as dump_file:
        json.dump(perturb_backward,dump_file,indent=4)

    os.system("./Machup.out "+forward_file)
    os.system("./Machup.out "+backward_file)

    return forward_file, backward_file

def run_trim_case(configs):

    configs["run"] = {"forces":""}
    del configs["solver"]["convergence"] 

    trim_file = "trim.json"
    with open(trim_file,'w') as dump_file:
        json.dump(configs,dump_file,indent=4)

    os.system("./Machup.out "+trim_file)

def get_forces_and_moments(filename,configs,r_CG):
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
            mom_trans = np.cross(r_CG,tail_FM[0])
            tail_FM[1,0] += coefs["total"][key]["Cl"]*q_inf*S_w*l_ref_lat-mom_trans[0]
            tail_FM[1,1] += coefs["total"][key]["Cm"]*q_inf*S_w*l_ref_lon-mom_trans[1]
            tail_FM[1,2] += coefs["total"][key]["Cn"]*q_inf*S_w*l_ref_lat-mom_trans[2]

    return lead_FM,tail_FM

def calc_cent_diff(configs,wrt,incr,r_CG):
    if wrt in "dxdydz":
        temp = run_perturbed_position(configs,wrt,incr)
    elif wrt in "dpdqdr":
        temp = run_perturbed_orientation(configs,wrt,incr)
    elif wrt in "aileronelevatorrudder":
        temp = run_perturbed_control(configs,wrt,incr)
    forward_file = temp[0]
    backward_file = temp[1]
    fwd_lead,fwd_tail = get_forces_and_moments(forward_file,configs,r_CG)
    bwd_lead,bwd_tail = get_forces_and_moments(backward_file,configs,r_CG)

    derivs = (fwd_tail-bwd_tail)/(2*incr)
    return derivs

if __name__=="__main__":

    # Find the matrix of derivatives of aerodynamic forces and moments with respect to position and orientation.

    # Initialize
    os.system("rm *forward*.json")
    os.system("rm *backward*.json")
    os.system("rm *trim*.json")

    trim_config_file = 'input.json'
    with open(trim_config_file) as trim_file:
        trim_configs = json.load(trim_file)

    run_trim_case(trim_configs)

    # Specify increments
    dx = 0.1
    dtheta = 0.1

    # Find displacement of trailing aircraft
    r_CG = np.zeros(3)
    r_CG[0] = trim_configs["trailer"]["CGx"]-trim_configs["plane"]["CGx"]
    r_CG[1] = trim_configs["trailer"]["CGy"]-trim_configs["plane"]["CGy"]
    r_CG[2] = trim_configs["trailer"]["CGz"]-trim_configs["plane"]["CGz"]

    # d/dx
    dFM_dx = calc_cent_diff(trim_configs,"dx",dx,r_CG)

    # d/dy
    dFM_dy = calc_cent_diff(trim_configs,"dy",dx,r_CG)

    # d/dz
    dFM_dz = calc_cent_diff(trim_configs,"dz",dx,r_CG)

    # d/dda
    dFM_dda = calc_cent_diff(trim_configs,"aileron",dtheta,r_CG)*180/np.pi

    # d/dde
    dFM_dde = calc_cent_diff(trim_configs,"elevator",dtheta,r_CG)*180/np.pi

    # d/ddr
    dFM_ddr = calc_cent_diff(trim_configs,"rudder",dtheta,r_CG)*180/np.pi

    print("\nDerivatives w.r.t. x:\n{0}".format(dFM_dx))
    print("\nDerivatives w.r.t. y:\n{0}".format(dFM_dy))
    print("\nDerivatives w.r.t. z:\n{0}".format(dFM_dz))

    print("\nDerivatives w.r.t. da:\n{0}".format(dFM_dda))
    print("\nDerivatives w.r.t. de:\n{0}".format(dFM_dde))
    print("\nDerivatives w.r.t. dr:\n{0}".format(dFM_ddr))
    
    F_pbar = np.asarray([dFM_dx.flatten(),dFM_dy.flatten(),dFM_dz.flatten()])
    print("\nFp:\n{0}".format(F_pbar))

    F_u = np.asarray([dFM_dda.flatten(),dFM_dde.flatten(),dFM_ddr.flatten()])
    print("\nFu:\n{0}".format(F_u))