import numpy as np
import machup.MU as MU
import matplotlib.pyplot as plt
import sys
import json
import scipy.optimize as opt
import polyFit as pf
import os

def analyze_airplane(filename,find_acs):
    # Initialize
    rho = 0.0020628 # slugs/ft^3
    g = 32.17

    airplane = MU.MachUp(filename)
    name = filename.replace(".json","")
    print("\n\n---"+name+"---")

    with open(filename) as json_file:
        configs = json.load(json_file)

    W = configs["info"]["weight"]
    E = configs["info"]["launch_energy"]

    # Determine launch performance
    Sw = airplane.Sw
    V = np.sqrt(2*E/(W/32.2))
    C_L_trim = W/(0.5*rho*V**2*Sw)

    b_w = configs["reference"]["lateral_length"]
    c = configs["reference"]["longitudinal_length"]
    R_A = b_w**2/Sw

    aero_state = {"V_mag":V,
                  "rho":rho,
                  "alpha":0,
                  "beta":0}

    FM = airplane.solve(aero_state=aero_state)
    derivs = airplane.derivatives(aero_state=aero_state)
    delta_x_CG = FM["MY"]/W
    
    print("\n---Launch State (alpha=0)---")
    print("Trim C_L: {0}".format(C_L_trim))
    print("Actual C_L: {0}".format(FM["FL"]/(0.5*rho*V**2*Sw)))
    print("C_m_a: {0}".format(derivs["stability_derivatives"]["Cm_a"]))
    print("C_l_B: {0}".format(derivs["stability_derivatives"]["Cl_b"]))
    print("C_n_B: {0}".format(derivs["stability_derivatives"]["Cn_b"]))
    print("Glide Ratio: {0}".format(FM["FL"]/FM["FD"]))
    print("Vel: {0} ft/s".format(V))
    print("Aero Pitching Moment: {0} lbf*ft".format(FM["MY"]))
    print("The CG should be moved {0} ft in x to be trim at launch".format(delta_x_CG))

    # Find trim state
    a0 = 0
    aero_state["alpha"] = a0
    M0 = airplane.solve(aero_state=aero_state)["MY"]

    a1 = 0.1
    approx_error = 1
    while approx_error>1e-6:
        aero_state["alpha"] = a1
        M1 = airplane.solve(aero_state=aero_state)["MY"]
        a2 = a1-M1*(a1-a0)/(M1-M0)
        approx_error = abs(a2-a1)

        a0 = a1
        M0 = M1
        a1 = a2

    a_trim = a2
    aero_state["alpha"] = a_trim
    C_L_trim = airplane.solve(aero_state=aero_state)["FL"]/(0.5*rho*aero_state["V_mag"]**2*Sw)
    V_trim = np.sqrt(W/(0.5*rho*Sw*C_L_trim))
    aero_state["V_mag"] = V_trim
    FM = airplane.solve(aero_state=aero_state)
    derivs = airplane.derivatives(aero_state=aero_state)

    print("\n---Trim State---")
    print("Trim C_L: {0}".format(C_L_trim))
    print("Alpha: {0}".format(a_trim))
    print("Cm,a: {0}".format(derivs["stability_derivatives"]["Cm_a"]))
    print("Cl,B: {0}".format(derivs["stability_derivatives"]["Cl_b"]))
    print("Cn,B: {0}".format(derivs["stability_derivatives"]["Cn_b"]))
    print("Glide Ratio: {0}".format(FM["FL"]/FM["FD"]))
    print("Vel: {0}".format(V_trim))

    # Plots
    num_points = 31
    a = np.linspace(a_trim-15,a_trim+15,num_points)
    C_L = np.zeros(num_points)
    C_D = np.zeros(num_points)
    C_m = np.zeros(num_points)
    L_D = np.zeros(num_points)
    P_R = np.zeros(num_points)
    static_margin = np.zeros(num_points)
    C_L_a = np.zeros(num_points)
    C_m_a = np.zeros(num_points)
    C_l_B = np.zeros(num_points)
    C_n_B = np.zeros(num_points)
    ac = np.zeros((3,num_points))

    for i,alpha in enumerate(a):
    	aero_state["alpha"] = alpha
    	FM = airplane.solve(aero_state=aero_state)
    	derivs = airplane.derivatives(aero_state=aero_state)
    	if find_acs: ac[:,i] = airplane.find_aero_center(aero_state=aero_state).flatten()

    	C_L[i] = FM["FL"]/(0.5*rho*V_trim**2*Sw)
    	C_D[i] = FM["FD"]/(0.5*rho*V_trim**2*Sw)
    	C_m[i] = FM["MY"]/(0.5*rho*V_trim**2*Sw*c)
    	L_D[i] = FM["FL"]/FM["FD"]

    	C_L_a[i] = derivs["stability_derivatives"]["CL_a"]
    	C_m_a[i] = derivs["stability_derivatives"]["Cm_a"]
    	C_l_B[i] = derivs["stability_derivatives"]["Cl_b"]
    	C_n_B[i] = derivs["stability_derivatives"]["Cn_b"]
    	static_margin[i] = -C_m_a[i]/C_L_a[i]

    # Power Calculations
    P_R = (C_D*W/np.power(C_L,1.5)*np.sqrt(2*W/Sw/rho))[np.where(C_L>0)]
    R_G0 = L_D
    V_c = -P_R/W

    data = {
        "a":a,
        "C_L":C_L,
        "C_D":C_D,
        "C_m":C_m,
        "L_D":L_D,
        "R_G0":R_G0,
        "P_R":P_R,
        "V_c":V_c,
        "C_L_a":C_L_a,
        "C_m_a":C_m_a,
        "C_l_B":C_l_B,
        "C_n_B":C_n_B,
        "static_margin":static_margin
    }

    if find_acs:
        data["ac"] = ac

    # Drag fit
    coefs,_ = pf.poly_fit(3,C_L,C_D)
    C_D0 = coefs[0]
    C_DL = coefs[1]
    e = 1/(np.pi*R_A*coefs[2])

    print("\n---C_D vs C_L---")
    print("C_D0: {0}".format(C_D0))
    print("C_DL: {0}".format(C_DL))
    print("e: {0}".format(e))

    # Velocities
    aero_state["alpha"] = 0
    stall_condition = airplane.stall_airspeed(W,aero_state=aero_state)
    V_MD = np.sqrt(2)/np.power(np.pi*e*R_A*C_D0,0.25)*np.sqrt(W/Sw/rho)
    V_MDV = 2/np.sqrt(np.pi*e*R_A*C_DL+np.sqrt((np.pi*e*R_A*C_DL)**2+12*np.pi*e*R_A*C_D0))*np.sqrt(W/Sw/rho)

    print("\n---Important Velocities---")
    print("V_stall: {0} ft/s".format(stall_condition["airspeed"]))
    print("V_MD: {0} ft/s".format(V_MD))
    print("V_MDV: {0} ft/s".format(V_MDV))

    # Placement of CG for trim at max glide ratio (i.e. V_MD)
    aero_state["V_mag"] = V_MD
    a_trim_MD = airplane.target_lift(W,aero_state=aero_state)["alpha"]
    aero_state["alpha"] = a_trim_MD
    FM = airplane.solve(aero_state=aero_state)
    delta_x_CG = FM["MY"]/W
    derivs = airplane.derivatives(aero_state=aero_state)

    print("\n---Max L/D State---")
    print("Trim C_L: {0}".format(W/(0.5*rho*V_MD**2*Sw)))
    print("Alpha trim: {0}".format(a_trim_MD))
    print("Cm,a: {0}".format(derivs["stability_derivatives"]["Cm_a"]))
    print("Cl,B: {0}".format(derivs["stability_derivatives"]["Cl_b"]))
    print("Cn,B: {0}".format(derivs["stability_derivatives"]["Cn_b"]))
    print("Glide Ratio: {0}".format(FM["FL"]/FM["FD"]))
    print("Vel: {0}".format(V_MD))
    print("Aero Pitching Moment: {0}".format(FM["MY"]))
    print("The CG should be moved {0} ft for trim at max L/D".format(delta_x_CG))

    # Simulator inputs
    aero_state["alpha"] = 0.0
    FM = airplane.solve(aero_state=aero_state)
    C_L_ref = FM["FL"]/(0.5*rho*Sw*aero_state["V_mag"]**2)
    V_ref = np.sqrt(W/(0.5*rho*Sw*C_L_ref))

    aero_state["V_mag"] = V_ref
    FM = airplane.solve(aero_state=aero_state)
    derivs = airplane.derivatives(aero_state=aero_state)
    CD = FM["FD"]/(0.5*rho*V_ref**2*Sw)
    coefs,_ = pf.poly_fit(3,(a*np.pi/180),C_D)

    Ixxb = configs["inertia"]["Ixxb"]
    Iyyb = configs["inertia"]["Iyyb"]
    Izzb = configs["inertia"]["Izzb"]
    Ixzb = configs["inertia"]["Ixzb"]
    Ixyb = configs["inertia"]["Ixyb"]
    Iyzb = configs["inertia"]["Iyzb"]

    hxb = 0.0
    hyb = 0.0
    hzb = 0.0

    sim_dict = {}

    sim_dict["aircraft"] = {"name":name}
    sim_dict["aircraft"]["wing_area"] = Sw
    sim_dict["aircraft"]["wing_span"] = b_w

    sim_dict["initial"] = {}
    sim_dict["initial"]["weight"] = configs["info"]["weight"]

    sim_dict["reference"] = {}
    sim_dict["reference"]["airspeed"] = V_ref
    sim_dict["reference"]["density"] = rho
    sim_dict["reference"]["elevator"] = 0.0
    sim_dict["reference"]["lift"] = FM["FL"]
    sim_dict["reference"]["Ixx"] = Ixxb
    sim_dict["reference"]["Iyy"] = Iyyb
    sim_dict["reference"]["Izz"] = Izzb
    sim_dict["reference"]["Ixy"] = Ixyb
    sim_dict["reference"]["Ixz"] = Ixzb
    sim_dict["reference"]["Iyz"] = Iyzb
    sim_dict["reference"]["hx"] = hxb
    sim_dict["reference"]["hy"] = hyb
    sim_dict["reference"]["hz"] = hzb
    sim_dict["reference"]["CD"] = CD
    sim_dict["reference"]["CL,a"] = derivs["stability_derivatives"]["CL_a"]
    sim_dict["reference"]["CD,a"] = coefs[1]
    sim_dict["reference"]["CD,a,a"] = coefs[2]
    sim_dict["reference"]["Cm,a"] = derivs["stability_derivatives"]["Cm_a"]
    sim_dict["reference"]["CY,b"] = derivs["stability_derivatives"]["CY_b"]
    sim_dict["reference"]["Cl,b"] = derivs["stability_derivatives"]["Cl_b"]
    sim_dict["reference"]["Cn,b"] = derivs["stability_derivatives"]["Cn_b"]
    sim_dict["reference"]["CL,q"] = derivs["damping_derivatives"]["CL_qbar"]
    sim_dict["reference"]["CD,q"] = derivs["damping_derivatives"]["CD_qbar"]
    sim_dict["reference"]["Cm,q"] = derivs["damping_derivatives"]["Cm_qbar"]
    sim_dict["reference"]["CY,p"] = derivs["damping_derivatives"]["CY_pbar"]
    sim_dict["reference"]["Cl,p"] = derivs["damping_derivatives"]["Cl_pbar"]
    sim_dict["reference"]["Cn,p"] = derivs["damping_derivatives"]["Cn_pbar"]
    sim_dict["reference"]["CY,r"] = derivs["damping_derivatives"]["CY_rbar"]
    sim_dict["reference"]["Cl,r"] = derivs["damping_derivatives"]["Cl_rbar"]
    sim_dict["reference"]["Cn,r"] = derivs["damping_derivatives"]["Cn_rbar"]

    sim_dict["reference"]["CL,de"] = 0.0
    sim_dict["reference"]["CD,de"] = 0.0
    sim_dict["reference"]["Cm,de"] = 0.0
    sim_dict["reference"]["CY,da"] = 0.0
    sim_dict["reference"]["Cl,da"] = 0.0
    sim_dict["reference"]["Cn,da"] = 0.0
    sim_dict["reference"]["CY,dr"] = 0.0
    sim_dict["reference"]["Cl,dr"] = 0.0
    sim_dict["reference"]["Cn,dr"] = 0.0
    sim_dict["reference"]["CD3"] = 0.0

    sim_json_filename = "7447-"+name+".json"
    with open(sim_json_filename,'w',newline='\r\n') as json_file:
        json.dump(sim_dict,json_file,indent=4)
    print("Sim file outputted to "+sim_json_filename)

    # Export file for Dynamic Analysis
    sim_dict["initial"]["density"] = rho
    sim_dict["initial"]["climb"] = 0.0
    sim_dict["initial"]["bank"] = 0.0
    sim_dict["initial"]["Mach_number"] = V_ref/994.85
    sim_dict["initial"]["T_0"] = 0.0
    sim_dict["initial"]["T^prime"] = 0.0

    sim_dict["reference"]["length"] = np.sqrt(Sw)

    CL = FM["FL"]/(0.5*rho*V_ref**2*Sw)
    Cm0 = FM["MY"]/(0.5*rho*V_ref**2*Sw*c)
    M = V_ref/994.85

    sim_dict["reference"]["Mach_number"] = M
    sim_dict["reference"]["weight"] = configs["info"]["weight"]

    sim_dict["reference"]["CL,ahat"] = 0.0
    sim_dict["reference"]["CD,ahat"] = 0.0
    sim_dict["reference"]["Cm,ahat"] = 0.0
    sim_dict["reference"]["CL,uhat"] = 0.0
    sim_dict["reference"]["CD,uhat"] = 0.0
    sim_dict["reference"]["Cm,uhat"] = 0.0

    sim_dict["reference"]["CD,M"] = 2*M/(1-M**2)*CD
    sim_dict["reference"]["CL,M"] = M/(1-M**2)*CL
    sim_dict["reference"]["Cm,M"] = M/(1-M**2)*Cm0

    sim_dict["reference"]["Cl,a"] = derivs["stability_derivatives"]["Cl_a"]
    sim_dict["reference"]["Cn,a"] = derivs["stability_derivatives"]["Cn_a"]
    sim_dict["reference"]["Cm,b"] = derivs["stability_derivatives"]["Cm_b"]

    dynamic_json_filename = name+"_Dynamic.json"
    with open(dynamic_json_filename,'w',newline='\r\n') as json_file:
        json.dump(sim_dict,json_file,indent=4)
    print("Dynamic file outputted to "+dynamic_json_filename)
    l_ref = np.sqrt(Sw)

    return data

def plot_data(data,find_acs,airplane_names,dir_name):

    if len(data) != len(airplane_names):
        raise ValueError("Data and names do not match")

    # Make plot directory
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    plt.rcParams["font.family"] = "serif"

    # C_L versus alpha
    plt.figure()
    for airplane_data in data:
        plt.plot(airplane_data["a"],airplane_data["C_L"])
    plt.xlabel("Angle of Attack [deg]")
    plt.ylabel("C_L")
    plt.legend(airplane_names)
    plt.savefig(dir_name+"/C_L")

    # C_D versus alpha
    plt.figure()
    for airplane_data in data:
        plt.plot(airplane_data["a"],airplane_data["C_D"])
    plt.xlabel("Angle of Attack [deg]")
    plt.ylabel("C_D")
    plt.legend(airplane_names)
    plt.savefig(dir_name+"/C_D")

    # C_D versus C_L
    plt.figure()
    for airplane_data in data:
        plt.plot(airplane_data["C_L"],airplane_data["C_D"])
    plt.xlabel("C_L")
    plt.ylabel("C_D")
    plt.legend(airplane_names)
    plt.savefig(dir_name+"/C_D_C_L")

    # C_m versus alpha
    plt.figure()
    for airplane_data in data:
        plt.plot(airplane_data["a"],airplane_data["C_m"])
    plt.xlabel("Angle of Attack [deg]")
    plt.ylabel("C_m")
    plt.legend(airplane_names)
    plt.savefig(dir_name+"/C_m")

    # L/D versus alpha
    plt.figure()
    for airplane_data in data:
        plt.plot(airplane_data["a"],airplane_data["L_D"])
    plt.xlabel("Angle of Attack [deg]")
    plt.ylabel("Lift to Drag Ratio")
    plt.legend(airplane_names)
    plt.savefig(dir_name+"/L_D")

    # P_R versus alpha
    plt.figure()
    for airplane_data in data:
        plt.plot(airplane_data["a"][np.where(airplane_data["C_L"]>0)],airplane_data["P_R"])
    plt.xlabel("Angle of Attack [deg]")
    plt.ylabel("Power Required [ft*lb/s]")
    plt.legend(airplane_names)
    plt.savefig(dir_name+"/P_R")

    # static_margin versus alpha
    plt.figure()
    for airplane_data in data:
        plt.plot(airplane_data["a"],airplane_data["static_margin"])
    plt.xlabel("Angle of Attack [deg]")
    plt.ylabel("Static Margin")
    plt.legend(airplane_names)
    plt.savefig(dir_name+"/static_margin")

    # R_G0 versus alpha
    plt.figure()
    for airplane_data in data:
        plt.plot(airplane_data["a"],airplane_data["R_G0"])
    plt.xlabel("Angle of Attack [deg]")
    plt.ylabel("No Wind Glide Ratio")
    plt.legend(airplane_names)
    plt.savefig(dir_name+"/R_G0")

    # V_c versus alpha
    plt.figure()
    for airplane_data in data:
        plt.plot(airplane_data["a"][np.where(airplane_data["C_L"]>0)],airplane_data["V_c"])
    plt.xlabel("Angle of Attack [deg]")
    plt.ylabel("Climb/Sink Rate [ft/s]")
    plt.legend(airplane_names)
    plt.savefig(dir_name+"/V_c")

    # C_m_a versus alpha
    plt.figure()
    for airplane_data in data:
        plt.plot(airplane_data["a"],airplane_data["C_m_a"])
    plt.xlabel("Angle of Attack [deg]")
    plt.ylabel("C_m,a")
    plt.legend(airplane_names)
    plt.savefig(dir_name+"/C_m_a")

    # C_l_B versus alpha
    plt.figure()
    for airplane_data in data:
        plt.plot(airplane_data["a"],airplane_data["C_l_B"])
    plt.xlabel("Angle of Attack [deg]")
    plt.ylabel("C_l,B")
    plt.legend(airplane_names)
    plt.savefig(dir_name+"/C_l_B")

    # C_n_B versus alpha
    plt.figure()
    for airplane_data in data:
        plt.plot(airplane_data["a"],airplane_data["C_n_B"])
    plt.xlabel("Angle of Attack [deg]")
    plt.ylabel("C_n,B")
    plt.legend(airplane_names)
    plt.savefig(dir_name+"/C_n_B")

    ## Locus of ac
    if find_acs:
        plt.figure()
        for airplane_data in data:
            plt.plot(airplane_data["ac"][0],airplane_data["ac"][2])
        plt.xlabel("x")
        plt.ylabel("z")
        plt.legend(airplane_names)
        plt.savefig(dir_name+"/ac")

    print("\nSuccessfully plotted aerdynamic data. Plots are stored in /"+plot_dir_name)

if __name__=="__main__":
    find_acs = False
    data = []
    names = []
    for airplane_file in sys.argv:
        if airplane_file == "StaticAnalysis.py":
            continue
        data.append(analyze_airplane(airplane_file,find_acs))
        names.append(airplane_file.replace(".json",""))

    plot_dir_name = "Plots"
    plot_data(data,find_acs,names,plot_dir_name)
