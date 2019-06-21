import numpy as np
import machup.MU as MU
import json
import scipy
import sys
import matplotlib.pyplot as plt

global g
g = 32.17

def get_airplane_dict(airplane_filename):
    with open(airplane_filename,'r') as airplane_file:
        airplane = json.load(airplane_file)
    return airplane

def solve_level_flight(a):
    print("\nSolving 12x12 dynamic system for "+a["aircraft"]["name"])
    Sw = a["aircraft"]["wing_area"]
    bw = a["aircraft"]["wing_span"]
    c = Sw/bw
    l_ref = a["reference"]["length"]
    rho = a["reference"]["density"]
    W = a["initial"]["weight"]
    V_ref = a["reference"]["airspeed"]

    CL = a["reference"]["lift"]/(0.5*rho*V_ref**2*Sw)
    CD = a["reference"]["CD"]
    CD_a = a["reference"]["CD,a"]
    CX_a = CL-CD_a
    CZ_a = -a["reference"]["CL,a"]-CD

    Ixxb = a["reference"]["Ixx"]
    Iyyb = a["reference"]["Iyy"]
    Izzb = a["reference"]["Izz"]
    Ixzb = a["reference"]["Ixy"]
    Ixyb = a["reference"]["Ixz"]
    Iyzb = a["reference"]["Iyz"]

    ixy = Ixyb/Ixxb
    ixz = Ixzb/Ixxb
    iyx = Ixyb/Iyyb
    iyz = Iyzb/Iyyb
    izx = Ixzb/Izzb
    izy = Iyzb/Izzb

    M = a["reference"]["Mach_number"]

    hxb = a["reference"]["hx"]
    hyb = a["reference"]["hy"]
    hzb = a["reference"]["hz"]

    CX0 = -CD
    CZ0 = -CL
    Cm0 = 0.0

    CX_muh = -a["reference"]["CL,uhat"]
    CZ_muh = -a["reference"]["CD,uhat"]
    Cm_muh = a["reference"]["Cm,uhat"]

    CX_ah = -a["reference"]["CD,ahat"]
    CZ_ah = -a["reference"]["CL,ahat"]
    Cm_ah = a["reference"]["Cm,ahat"]

    CX_mu = -M*a["reference"]["CD,M"]
    CZ_mu = -M*a["reference"]["CL,M"]
    Cm_mu = M*a["reference"]["Cm,M"]

    Cl_a = a["reference"]["Cl,a"]
    Cm_a = a["reference"]["Cm,a"]
    Cn_a = a["reference"]["Cn,a"]

    CY_B = a["reference"]["CY,b"]
    Cl_B = a["reference"]["Cl,b"]
    Cm_B = a["reference"]["Cm,b"]
    Cn_B = a["reference"]["Cn,b"]

    CY_pbar = a["reference"]["CY,p"]
    Cl_pbar = a["reference"]["Cl,p"]
    Cn_pbar = a["reference"]["Cn,p"]

    CX_qbar = -a["reference"]["CD,q"]
    CZ_qbar = -a["reference"]["CL,q"]
    Cm_qbar = a["reference"]["Cm,q"]

    CY_rbar = a["reference"]["CY,r"]
    Cl_rbar = a["reference"]["Cl,r"]
    Cn_rbar = a["reference"]["Cn,r"]

    B = np.eye(12)
    B_non_dim = rho*Sw*c/(4*W/g)
    B_mom_non_dim = rho*Sw*c**2*l_ref/(4*Iyyb)

    B[0,0] -= B_non_dim*CX_muh
    B[0,2] -= B_non_dim*CX_ah

    B[2,0] -= B_non_dim*CZ_muh
    B[2,2] -= B_non_dim*CZ_ah

    B[4,0] -= B_mom_non_dim*Cm_muh
    B[4,2] -= B_mom_non_dim*Cm_ah

    B[3,4] -= ixy
    B[3,5] -= ixz

    B[4,3] -= iyx
    B[4,5] -= iyz

    B[5,3] -= izx
    B[5,4] -= izy
    
    phi0 = a["initial"]["bank"]
    theta0 = a["initial"]["climb"]

    Sphi0 = np.sin(np.radians(phi0))
    Cphi0 = np.cos(np.radians(phi0))
    Tphi0 = np.tan(np.radians(phi0))
    Ctheta0 = np.cos(np.radians(theta0))
    Stheta0 = np.sin(np.radians(theta0))
    Ttheta0 = np.tan(np.radians(theta0))
    A = np.zeros((12,12))
    Ag = g*l_ref/V_ref**2

    A[0,0] = rho*Sw*l_ref/(2*W/g)*(2*CX0+CX_mu)
    A[0,1] = Ag*Sphi0*Ctheta0
    A[0,2] = rho*Sw*l_ref/(2*W/g)*CX_a-Ag*Tphi0*Sphi0*Ctheta0
    A[0,4] = rho*Sw*c/(4*W/g)*CX_qbar
    A[0,10] = -Ag*Ctheta0

    A[1,0] = -Ag*Sphi0*Ctheta0
    A[1,1] = rho*Sw*l_ref/(2*W/g)*CY_B
    A[1,2] = -Ag*Tphi0*Stheta0
    A[1,3] = rho*Sw*bw/(4*W/g)*CY_pbar
    A[1,5] = rho*Sw*bw/(4*W/g)*CY_rbar-1
    A[1,9] = Ag*Cphi0*Ctheta0
    A[1,10] = -Ag*Stheta0*Sphi0

    A[2,0] = rho*Sw*l_ref/(2*W/g)*(2*CZ0+CZ_mu)+Ag*Tphi0*Sphi0*Ctheta0
    A[2,1] = Ag*Tphi0*Stheta0
    A[2,2] = rho*Sw*l_ref/(2*W/g)*CZ_a
    A[2,4] = rho*Sw*c/(4*W/g)*CZ_qbar+1
    A[2,9] = -Ag*Sphi0*Ctheta0
    A[2,10] = -Ag*Cphi0*Stheta0

    A[3,1] = rho*Sw*bw*l_ref**2/(2*Ixxb)*Cl_B
    A[3,2] = rho*Sw*bw*l_ref**2/(2*Ixxb)*Cl_a
    A[3,3] = rho*Sw*bw**2*l_ref/(4*Ixxb)*Cl_pbar+Ag*(Ixzb*Tphi0*Sphi0*Ctheta0-Ixyb*Sphi0*Ctheta0)/Ixxb
    A[3,4] = -(hzb*l_ref/(Ixxb*V_ref)+Ag*((Izzb-Iyyb)*Sphi0*Ctheta0-2*Iyzb*Tphi0*Sphi0*Ctheta0+Ixzb*Tphi0*Stheta0)/Ixxb)
    A[3,5] = rho*Sw*bw**2*l_ref/(4*Ixxb)*Cl_rbar+(hyb*l_ref/(Ixxb*V_ref)+Ag*((Iyyb-Izzb)*Tphi0*Sphi0*Ctheta0-2*Ixyb*Sphi0*Ctheta0+Ixyb*Tphi0*Stheta0)/Ixxb)
    
    A[4,0] = rho*Sw*c*l_ref**2/(2*Iyyb)*(2*Cm0+Cm_mu)
    A[4,1] = rho*Sw*c*l_ref**2/(2*Iyyb)*Cm_B
    A[4,2] = rho*Sw*c*l_ref**2/(2*Iyyb)*Cm_a
    A[4,3] = hzb*l_ref/(Iyyb*V_ref)+Ag*((Izzb-Ixxb)*Sphi0*Ctheta0+2*Ixzb*Tphi0*Stheta0-Iyzb*Tphi0*Sphi0*Ctheta0)/Iyyb
    A[4,4] = rho*Sw*c**2*l_ref/(4*Iyyb)*Cm_qbar+Ag*(Ixyb*Sphi0*Ctheta0+Iyzb*Tphi0*Stheta0)/Iyyb
    A[4,5] = -(hxb*l_ref/(Iyyb*V_ref)+Ag*((Izzb-Ixxb)*Tphi0*Stheta0-2*Ixzb*Sphi0*Ctheta0-Ixyb*Tphi0*Sphi0*Ctheta0)/Iyyb)

    A[5,1] = rho*Sw*bw*l_ref**2/(2*Izzb)*Cn_B
    A[5,2] = rho*Sw*bw*l_ref**2/(2*Izzb)*Cn_a
    A[5,3] = rho*Sw*bw**2*l_ref/(4*Izzb)*Cn_pbar-(hyb*l_ref/(Izzb*V_ref)+Ag*((Iyyb-Ixxb)*Tphi0*Sphi0*Ctheta0+2*Ixyb*Tphi0*Stheta0-Iyzb*Sphi0*Ctheta0)/Izzb)
    A[5,4] = hxb*l_ref/(Izzb*V_ref)+Ag*((Iyyb-Ixxb)*Tphi0*Stheta0-2*Ixyb*Tphi0*Sphi0*Ctheta0-Ixzb*Sphi0*Ctheta0)/Izzb
    A[5,5] = rho*Sw*bw**2*l_ref/(4*Izzb)*Cn_rbar+Ag*(-Iyzb*Tphi0*Stheta0-Ixzb*Tphi0*Sphi0*Ctheta0)/Izzb

    A[6,0] = Ctheta0
    A[6,1] = Sphi0*Stheta0
    A[6,2] = Cphi0*Stheta0
    A[6,10] = -Stheta0

    A[7,1] = Cphi0
    A[7,2] = -Sphi0
    A[7,11] = Ctheta0

    A[8,0] = -Stheta0
    A[8,1] = Sphi0*Ctheta0
    A[8,2] = Cphi0*Ctheta0
    A[8,10] = -Ctheta0

    A[9,3] = 1
    A[9,4] = Sphi0*Ttheta0
    A[9,10] = Ag*Tphi0/Ctheta0

    A[10,4] = Cphi0
    A[10,5] = -Sphi0
    A[10,9] = -Ag*Tphi0*Ctheta0

    A[11,4] = Sphi0/Ctheta0
    A[11,5] = Cphi0/Ctheta0
    A[11,10] = Ag*Tphi0*Ttheta0

    (vals,vecs) = np.linalg.eig(np.linalg.inv(B)*np.matrix(A))
    vecs = vecs.T

    # Acceleration sensitivity
    CW = W/(0.5*rho*V_ref**2*Sw)
    AS = 1/CW*a["reference"]["CL,a"]
    print("Acceleration Sensitivity: {0}".format(AS))

    # Overdamped modes
    od = np.where((abs(np.real(vals))>1e-12) & (abs(np.imag(vals))<1e-12))
    od = list(od[0])

    # Roll Mode
    ir = np.argmax(abs(vals[od]))
    roll_eigenval = vals[od][ir]
    report_eigendata(vals[od][ir],vecs[od][ir],"Roll Mode",V_ref,l_ref)

    # Spiral Mode
    isp = np.argmin(abs(vals[od]))
    spiral_eigenval = vals[od][isp]
    report_eigendata(vals[od][isp],vecs[od][isp],"Spiral Mode",V_ref,l_ref)

    # Periodic modes
    periodic = np.where((np.imag(vals)>1e-12))
    periodic = list(periodic[0])

    # Dutch Roll Mode
    dr = np.argmax(abs(vecs[periodic,1]))
    dutch_eigenval = vals[periodic][dr]
    report_eigendata(vals[periodic][dr],vecs[periodic][dr],"Dutch Roll",V_ref,l_ref)

    # Long Period Mode
    lp = np.argmin(abs(vals[periodic]))
    phugoid_eigenval = vals[periodic][lp]
    report_eigendata(vals[periodic][lp],vecs[periodic][lp],"Phugoid",V_ref,l_ref)

    if len(periodic)==2: # Short period is overdamped
        if isp>ir:
            del od[isp]
            del od[ir]
        else:
            del od[ir]
            del od[isp]
        short_eigenval = (vals[od[0]],vals[od[1]])
        report_overdamped_mode(vals[od[0]],vals[od[1]],vecs[od[0]],vecs[od[1]],"Short Period",V_ref,l_ref)
    else:
        if dr>lp:
            del periodic[dr]
            del periodic[lp]
        else:
            del periodic[lp]
            del periodic[dr]
        short_eigenval = vals[periodic[0]]
        report_eigendata(vals[periodic[0]],vecs[periodic[0]],"Short Period",V_ref,l_ref)

    return AS,roll_eigenval,spiral_eigenval,dutch_eigenval,phugoid_eigenval,short_eigenval

def report_eigendata(val,vec,name,V_ref,l_ref):
    print("\n---"+name+"---")

    # Value
    print("Lambda: {0}".format(val))
    r = np.asscalar(np.real(val))
    i = np.asscalar(np.imag(val))
    mag = np.sqrt(r**2+i**2)
    sigma = -r*V_ref/l_ref
    print("Damping Rate: {0}".format(sigma))
    if sigma>0:
        print("99% Damping Time: {0}".format(-np.log(0.01)/sigma))
    else:
        print("Doubling Time: {0}".format(-np.log(2)/sigma))

    if abs(i)>1e-12: # Complex mode
        zeta = -(2*r)/(2*mag)
        print("Damping Ratio: {0}".format(zeta))
        wn = mag*V_ref/l_ref
        wd = abs(i)*V_ref/l_ref
        print("Natural Frequency: {0}".format(wn))
        print("Period: {0}".format(2*np.pi/wd))

    # Vector
    print("{0:<50}, {1:<30}, {2:<20}".format("Vector","Amplitude","Phase"))
    for i in range(12):
        r_c = np.real(np.asscalar(vec[:,i]))
        i_c = np.imag(np.asscalar(vec[:,i]))
        A = np.sqrt(r_c**2+i_c**2)
        theta = np.degrees(np.arctan2(i_c,r_c))
        print("{0:<50}, {1:<30}, {2:<20}".format(np.asscalar(vec[:,i]),A,theta))


def report_overdamped_mode(l0,l1,X0,X1,name,V_ref,l_ref):
    print("\n---"+name+"---")
    zeta = -(l0+l1)/(2*np.sqrt(l0*l1))
    print("Lambda 0: {0}".format(l0))
    print("Lambda 1: {0}".format(l1))
    print("Damping Ratio: {0}".format(zeta))
    print("Natural Frequency: {0}".format(np.sqrt(l0*l1)*V_ref/l_ref))
    print("{0:<50}, {1:<30}, {2:<20}".format("Vector","Amplitude","Phase"))
    print("Chi 0:")
    for i in range(12):
        r_c = np.real(np.asscalar(X0[:,i]))
        i_c = np.imag(np.asscalar(X0[:,i]))
        A = np.sqrt(r_c**2+i_c**2)
        theta = np.degrees(np.arctan2(i_c,r_c))
        print("{0:<50}, {1:<30}, {2:<20}".format(np.asscalar(X0[:,i]),A,theta))
    print("Chi 1:")
    for i in range(12):
        r_c = np.real(np.asscalar(X1[:,i]))
        i_c = np.imag(np.asscalar(X1[:,i]))
        A = np.sqrt(r_c**2+i_c**2)
        theta = np.degrees(np.arctan2(i_c,r_c))
        print("{0:<50}, {1:<30}, {2:<20}".format(np.asscalar(X1[:,i]),A,theta))


if __name__ == "__main__":

    trim_vel = []
    stall_vel = [16.06,13.17]
    AS = []
    eigenvals = []
    l_ref = []
    names = []
    num_airplanes = 0
    for airplane_file in sys.argv:
        if airplane_file == "DynamicAnalysis.py":
            continue
        num_airplanes += 1
        a = get_airplane_dict(airplane_file)

        trim_vel.append(a["reference"]["airspeed"])
        l_ref.append(a["reference"]["length"])
        names.append(a["aircraft"]["name"])

        return_vals = solve_level_flight(a)
        AS.append(return_vals[0])
        eigenvals.append(return_vals[1:])
    
    # Plots
    dir_name = "Plots"
    plt.rcParams["font.family"] = "serif"

    # Short period damped natural frequency
    plt.figure()
    for i in range(num_airplanes):
        vel_space = np.linspace(stall_vel[i],2*trim_vel[i],100)
        try:
            wn = abs(np.imag(eigenvals[i][4]))*vel_space/l_ref[i]
            plt.plot(vel_space,wn)
        except:
            wn = np.sqrt(np.real(eigenvals[i][4][0])**2+np.real(eigenvals[i][4][1])**2)*vel_space/l_ref[i]
            plt.plot(vel_space,wn)
    plt.xlabel("Velocity [ft/s]")
    plt.ylabel("Short Period Damped Frequency [rad/s]")
    plt.legend(names)
    plt.savefig(dir_name+"/ShortPeriodWd")

    # Short period damping ratio
    plt.figure()
    for i in range(num_airplanes):
        vel_space = np.linspace(stall_vel[i],2*trim_vel[i],100)
        try:
            plt.plot(vel_space,np.full(100,-(2*np.real(eigenvals[i][4])/(2*np.sqrt(np.real(eigenvals[i][4])**2+np.imag(eigenvals[i][4])**2)))))
        except:
            plt.plot(vel_space,np.full(100,-(np.real(eigenvals[i][4][0])+np.real(eigenvals[i][4][1]))/(2*np.sqrt(np.real(eigenvals[i][4][0])*np.real(eigenvals[i][4][1])))))
    plt.xlabel("Velocity [ft/s]")
    plt.ylabel("Short Period Damping Ratio")
    plt.legend(names)
    plt.savefig(dir_name+"/ShortPeriodZeta")

    # CAP
    plt.figure()
    for i in range(num_airplanes):
        vel_space = np.linspace(stall_vel[i],2*trim_vel[i],100)
        plt.plot(vel_space,wn/AS[i])
    plt.xlabel("Velocity [ft/s]")
    plt.ylabel("CAP [(rad/s)^2/(g/rad)]")
    plt.legend(names)
    plt.savefig(dir_name+"/CAP")

    # Phugoid damped natural frequency
    plt.figure()
    for i in range(num_airplanes):
        vel_space = np.linspace(stall_vel[i],2*trim_vel[i],100)
        plt.plot(vel_space,abs(np.imag(eigenvals[i][3]))*vel_space/l_ref[i])
    plt.xlabel("Velocity [ft/s]")
    plt.ylabel("Phugoid Damped Frequency [rad/s]")
    plt.legend(names)
    plt.savefig(dir_name+"/PhugoidWd")

    # Phugoid damping ratio
    plt.figure()
    for i in range(num_airplanes):
        vel_space = np.linspace(stall_vel[i],2*trim_vel[i],100)
        plt.plot(vel_space,np.full(100,-(2*np.real(eigenvals[i][3])/(2*np.sqrt(np.real(eigenvals[i][3])**2+np.imag(eigenvals[i][3])**2)))))
    plt.xlabel("Velocity [ft/s]")
    plt.ylabel("Phugoid Damping Ratio")
    plt.legend(names)
    plt.savefig(dir_name+"/PhugoidZeta")

    # Dutch roll damped natural frequency
    plt.figure()
    for i in range(num_airplanes):
        vel_space = np.linspace(stall_vel[i],2*trim_vel[i],100)
        plt.plot(vel_space,abs(np.imag(eigenvals[i][2]))*vel_space/l_ref[i])
    plt.xlabel("Velocity [ft/s]")
    plt.ylabel("Dutch Roll Damped Frequency [rad/s]")
    plt.legend(names)
    plt.savefig(dir_name+"/DutchRollWd")

    # Dutch roll damping ratio
    plt.figure()
    for i in range(num_airplanes):
        vel_space = np.linspace(stall_vel[i],2*trim_vel[i],100)
        plt.plot(vel_space,np.full(100,-(2*np.real(eigenvals[i][2])/(2*np.sqrt(np.real(eigenvals[i][2])**2+np.imag(eigenvals[i][2])**2)))))
    plt.xlabel("Velocity [ft/s]")
    plt.ylabel("Dutch Roll Damping Ratio")
    plt.legend(names)
    plt.savefig(dir_name+"/DutchRollZeta")

    # Roll time constant
    plt.figure()
    for i in range(num_airplanes):
        vel_space = np.linspace(stall_vel[i],2*trim_vel[i],100)
        plt.plot(vel_space,1/-np.real(eigenvals[i][0])*l_ref[i]/vel_space)
    plt.xlabel("Velocity [ft/s]")
    plt.ylabel("Roll Time Constant [s]")
    plt.legend(names)
    plt.savefig(dir_name+"/RollTau")

    # Spiral 99% damping time or time to double
    plt.figure()
    for i in range(num_airplanes):
        vel_space = np.linspace(stall_vel[i],2*trim_vel[i],100)
        plt.plot(vel_space,np.log(2)/(np.real(eigenvals[i][1])*vel_space/l_ref[i]))
    plt.xlabel("Velocity [ft/s]")
    plt.ylabel("Spiral Doubling Time [s]")
    plt.legend(names)
    plt.savefig(dir_name+"/SpiralTime")
