import numpy as np
import json
import os
import copy
import scipy.optimize as opt
import matplotlib.pyplot as plt

class TailingScenario:
    """Defines the situation of one aircraft trailing another. Separation is referenced from the leading aircraft in aircraft coordinates."""

    def __init__(self,lead_filename,tail_filename,**kwargs):
        self.lead_filename = lead_filename
        self.tail_filename = tail_filename
        self.dx = kwargs.get("dx",0.1)
        self.dtheta = kwargs.get("dtheta",0.1)
        self.grid = kwargs.get("grid",40)

        self._combine()

    def _combine(self):
        """Combines two plane .json files."""

        # Load initial files
        with open(self.lead_filename,'r') as lead_file_handle:
            self.lead_dict = json.load(lead_file_handle)

        with open(self.tail_filename,'r') as tail_file_handle:
            self.tail_dict = json.load(tail_file_handle)

        # Create new copy
        self.combined_dict = copy.deepcopy(self.lead_dict)

        # Alter some key values
        del self.combined_dict["solver"]["convergence"] # Use default convergence to avoid FORTRAN not being able to read Python-exported floating points
        self.combined_dict["controls"] = {}
        self.combined_dict["wings"] = {}
        self.combined_dict["airfoil_DB"] = "./AirfoilDatabase"
        self.combined_dict["run"] = {"forces":""}
        self.combined_dict["plane"]["name"] = "lead_tail_combo"
        self.combined_dict["condition"]["alpha"] = 0.0
        self.combined_dict["condition"]["beta"] = 0.0

        # Copy weights
        del self.combined_dict["condition"]["W"]
        self.combined_dict["condition"]["W_lead"] = copy.copy(self.lead_dict["condition"]["W"])
        self.combined_dict["condition"]["W_tail"] = copy.copy(self.tail_dict["condition"]["W"])

        # Copy lead controls
        for key in self.lead_dict["controls"]:
            new_key = "lead_"+key
            self.combined_dict["controls"][new_key] = copy.copy(self.lead_dict["controls"][key])

        # Copy lead surfaces
        for key in self.lead_dict["wings"]:
            new_key = "lead_"+key
            self.combined_dict["wings"][new_key] = copy.copy(self.lead_dict["wings"][key])
            self.combined_dict["wings"][new_key]["grid"] = self.grid
            for control_key in self.combined_dict["wings"][new_key]["control"]["mix"]:
                self.combined_dict["wings"][new_key]["control"]["mix"] = {"lead_"+control_key:1.0}

        # Copy tail controls
        for key in self.tail_dict["controls"]:
            new_key = "tail_"+key
            self.combined_dict["controls"][new_key] = copy.copy(self.tail_dict["controls"][key])

        # For tailing lifting surfaces, split double surfaces into two surfaces
        for key in self.tail_dict["wings"]:
            new_key = "tail_"+key

            if self.tail_dict["wings"][key]["side"] == "both": # Double sided surface
                # Left side
                self.combined_dict["wings"][new_key+"_left"] = copy.deepcopy(self.tail_dict["wings"][key])
                self.combined_dict["wings"][new_key+"_left"]["side"] = "left"
                for control_key in self.combined_dict["wings"][new_key+"_left"]["control"]["mix"]:
                    self.combined_dict["wings"][new_key+"_left"]["control"]["mix"] = {"tail_"+control_key:1.0}
                    
                # Right Side
                self.combined_dict["wings"][new_key+"_right"] = copy.deepcopy(self.tail_dict["wings"][key])
                self.combined_dict["wings"][new_key+"_right"]["side"] = "right"
                for control_key in self.combined_dict["wings"][new_key+"_right"]["control"]["mix"]:
                    self.combined_dict["wings"][new_key+"_right"]["control"]["mix"] = {"tail_"+control_key:1.0}

            else: # Single sided surface
                self.combined_dict["wings"][new_key] = self.tail_dict["wings"][key]
                for control_key in self.combined_dict["wings"][new_key]["control"]["mix"]:
                    self.combined_dict["wings"][new_key]["control"]["mix"] = {"tail_"+control_key:1.0}

        # Apply grid resolution to all lifting surfaces
        for wing in self.combined_dict["wings"]:
            self.combined_dict["wings"][wing]["grid"] = self.grid

        # Create new .json file
        self.untrimmed_filename = "combined_untrimmed.json"
        with open(self.untrimmed_filename,'w') as dump_file_handle:
            json.dump(self.combined_dict,dump_file_handle,indent=4)
        
    def _apply_alpha_de(self,alpha,de,airplane):
        """Applies an angle of attack and elevator deflection to the specified airplane."""

        self.trimmed_dict["controls"][airplane+"_elevator"]["deflection"] = de

        for wing in self.trimmed_dict["wings"]:
            if airplane in wing:
                x0 = copy.copy(self.combined_dict["wings"][wing]["connect"]["dx"])
                z0 = copy.copy(self.combined_dict["wings"][wing]["connect"]["dz"])
                C = np.cos(np.radians(alpha))
                S = np.sin(np.radians(alpha))
                x1 = x0*C+z0*S
                z1 = z0*C-x0*S
                if airplane == "tail":
                    x1 += self.r[0]
                    z1 += self.r[2]
                self.trimmed_dict["wings"][wing]["connect"]["dx"] = copy.copy(x1)
                self.trimmed_dict["wings"][wing]["connect"]["dz"] = copy.copy(z1)

                if "v" in wing: # Vertical stabilizer
                    self.trimmed_dict["wings"][wing]["sweep"] = self.combined_dict["wings"][wing]["sweep"]+alpha
                else: # Horizontal surface
                    self.trimmed_dict["wings"][wing]["mounting_angle"] = self.combined_dict["wings"][wing]["mounting_angle"]+alpha
        
        with open(self.trimmed_filename,'w') as dump_file_handle:
            json.dump(self.trimmed_dict,dump_file_handle,indent=4)

    def _get_trim_residuals(self,x,airplane):
        """Returns L-W and m for the given aircraft in the given state."""
        alpha,de = x
        self._apply_alpha_de(alpha,de,airplane)
        os.system("./Machup.out "+self.trimmed_filename)
        lead_FM,tail_FM = self.get_forces_and_moments(self.trimmed_filename)
        if airplane == "tail":
            R_L = np.asscalar(-tail_FM[0,2])-self.combined_dict["condition"]["W_"+airplane]
            R_m = np.asscalar(tail_FM[1,1])
        else:
            R_L = np.asscalar(-lead_FM[0,2])-self.combined_dict["condition"]["W_"+airplane]
            R_m = np.asscalar(lead_FM[1,1])
        return R_L,R_m

    def apply_separation_and_trim(self,separation_vec,iterations=1):
        """Separates the two aircraft according to separation_vec and trims both.
        This will leave alpha and beta as 0.0, instead trimming using mounting
        angles. This will first trim the leading aircraft, then the tailing aircraft.
        This process can be repeated for finer trim results using the iterations
        parameter."""
        self.r = separation_vec
        self.trimmed_dict = copy.deepcopy(self.combined_dict)
        for wing in self.trimmed_dict["wings"]:
            if "tail" in wing:
                self.trimmed_dict["wings"][wing]["connect"]["dx"] += self.r[0]
                self.trimmed_dict["wings"][wing]["connect"]["dy"] += self.r[1]
                self.trimmed_dict["wings"][wing]["connect"]["dz"] += self.r[2]
        
        self.trimmed_filename = "trimmed.json"
        with open(self.trimmed_filename,'w') as dump_file_handle:
            json.dump(self.trimmed_dict,dump_file_handle,indent=4)

        for _ in range(iterations):
            airplanes = ["lead","tail"]
            for airplane in airplanes:
                if airplane == "lead":
                    alpha_init = self.lead_dict["condition"]["alpha"]
                else:
                    alpha_init = self.tail_dict["condition"]["alpha"]

                de_init = self.trimmed_dict["controls"][airplane+"_elevator"]["deflection"]

                _ = opt.root(self._get_trim_residuals,[alpha_init,de_init],args=(airplane,))

    def _run_perturbation(self,wrt):
        """Perturbs the aircraft with respect to wrt."""
        perturb_forward = copy.deepcopy(self.trimmed_dict)
        perturb_backward = copy.deepcopy(self.trimmed_dict)

        if wrt in "dxdydz": # Position
            for key in self.trimmed_dict["wings"]:
                if "tail" in key:
                    perturb_forward["wings"][key]["connect"][wrt] = self.trimmed_dict["wings"][key]["connect"][wrt]+self.dx
                    perturb_backward["wings"][key]["connect"][wrt] = self.trimmed_dict["wings"][key]["connect"][wrt]-self.dx

        elif wrt in "aileronelevatorrudder": # Control input
            control_key = "tail_"+wrt

            perturb_forward["controls"][control_key]["deflection"] = self.trimmed_dict["controls"][control_key]["deflection"]+self.dtheta
            perturb_backward["controls"][control_key]["deflection"] = self.trimmed_dict["controls"][control_key]["deflection"]-self.dtheta
        
        elif wrt in "dpdqdr": # Orientation
            for wing in self.trimmed_dict["wings"]:
                if "tail" in wing:
                    if wrt == "dq": # Pitch
                        # Perturb forward
                        x0 = copy.copy(self.trimmed_dict["wings"][wing]["connect"]["dx"])
                        z0 = copy.copy(self.trimmed_dict["wings"][wing]["connect"]["dz"])
                        x0 -= self.r[0]
                        z0 -= self.r[2]
                        C = np.cos(np.radians(self.dtheta))
                        S = np.sin(np.radians(self.dtheta))
                        x1 = x0*C+z0*S
                        z1 = z0*C-x0*S
                        x1 += self.r[0]
                        z1 += self.r[2]
                        perturb_forward["wings"][wing]["connect"]["dx"] = copy.copy(x1)
                        perturb_forward["wings"][wing]["connect"]["dz"] = copy.copy(z1)

                        if "v" in wing: # Vertical stabilizer
                            perturb_forward["wings"][wing]["sweep"] += self.dtheta
                        else: # Horizontal surface
                            perturb_forward["wings"][wing]["mounting_angle"] += self.dtheta

                        # Perturb backward
                        x0 = copy.copy(self.trimmed_dict["wings"][wing]["connect"]["dx"])
                        z0 = copy.copy(self.trimmed_dict["wings"][wing]["connect"]["dz"])
                        x0 -= self.r[0]
                        z0 -= self.r[2]
                        C = np.cos(np.radians(-self.dtheta))
                        S = np.sin(np.radians(-self.dtheta))
                        x1 = x0*C+z0*S
                        z1 = z0*C-x0*S
                        x1 += self.r[0]
                        z1 += self.r[2]
                        perturb_backward["wings"][wing]["connect"]["dx"] = copy.copy(x1)
                        perturb_backward["wings"][wing]["connect"]["dz"] = copy.copy(z1)

                        if "v" in wing: # Vertical stabilizer
                            perturb_backward["wings"][wing]["sweep"] -= self.dtheta
                        else: # Horizontal surface
                            perturb_backward["wings"][wing]["mounting_angle"] -= self.dtheta

                    elif wrt == "dp": # Roll
                        # Perturb forward
                        y0 = copy.copy(self.trimmed_dict["wings"][wing]["connect"]["dy"])
                        z0 = copy.copy(self.trimmed_dict["wings"][wing]["connect"]["dz"])
                        y0 -= self.r[1]
                        z0 -= self.r[2]
                        C = np.cos(np.radians(self.dtheta))
                        S = np.sin(np.radians(self.dtheta))
                        y1 = y0*C-z0*S
                        z1 = z0*C+y0*S
                        y1 += self.r[1]
                        z1 += self.r[2]
                        perturb_forward["wings"][wing]["connect"]["dy"] = copy.copy(y1)
                        perturb_forward["wings"][wing]["connect"]["dz"] = copy.copy(z1)

                        if self.trimmed_dict["wings"][wing]["side"] == "right":
                            perturb_forward["wings"][wing]["dihedral"] -= self.dtheta
                        else:
                            perturb_forward["wings"][wing]["dihedral"] += self.dtheta

                        # Perturb backward
                        y0 = copy.copy(self.trimmed_dict["wings"][wing]["connect"]["dy"])
                        z0 = copy.copy(self.trimmed_dict["wings"][wing]["connect"]["dz"])
                        y0 -= self.r[1]
                        z0 -= self.r[2]
                        C = np.cos(np.radians(-self.dtheta))
                        S = np.sin(np.radians(-self.dtheta))
                        y1 = y0*C-z0*S
                        z1 = z0*C+y0*S
                        y1 += self.r[1]
                        z1 += self.r[2]
                        perturb_backward["wings"][wing]["connect"]["dy"] = copy.copy(y1)
                        perturb_backward["wings"][wing]["connect"]["dz"] = copy.copy(z1)

                        if self.trimmed_dict["wings"][wing]["side"] == "right":
                            perturb_backward["wings"][wing]["dihedral"] += self.dtheta
                        else:
                            perturb_backward["wings"][wing]["dihedral"] -= self.dtheta

                    elif wrt == "dr": # Yaw
                        # Perturb forward
                        x0 = copy.copy(self.trimmed_dict["wings"][wing]["connect"]["dx"])
                        y0 = copy.copy(self.trimmed_dict["wings"][wing]["connect"]["dy"])
                        x0 -= self.r[0]
                        y0 -= self.r[1]
                        C = np.cos(np.radians(self.dtheta))
                        S = np.sin(np.radians(self.dtheta))
                        x1 = x0*C-y0*S
                        y1 = y0*C+x0*S
                        x1 += self.r[0]
                        y1 += self.r[1]
                        perturb_forward["wings"][wing]["connect"]["dx"] = copy.copy(x1)
                        perturb_forward["wings"][wing]["connect"]["dy"] = copy.copy(y1)

                        if "v" in wing: # Vertical stabilizer
                            if perturb_forward["wings"][wing]["side"] == "right":
                                perturb_forward["wings"][wing]["mounting_angle"] -= self.dtheta
                            else:
                                perturb_forward["wings"][wing]["mounting_angle"] += self.dtheta
                        else: # Horizontal surface
                            if perturb_forward["wings"][wing]["side"] == "right":
                                perturb_forward["wings"][wing]["sweep"] += self.dtheta
                            else:
                                perturb_forward["wings"][wing]["sweep"] -= self.dtheta

                        # Perturb backward
                        x0 = copy.copy(self.trimmed_dict["wings"][wing]["connect"]["dx"])
                        y0 = copy.copy(self.trimmed_dict["wings"][wing]["connect"]["dy"])
                        x0 -= self.r[0]
                        y0 -= self.r[1]
                        C = np.cos(np.radians(-self.dtheta))
                        S = np.sin(np.radians(-self.dtheta))
                        x1 = x0*C-y0*S
                        y1 = y0*C+x0*S
                        x1 += self.r[0]
                        y1 += self.r[1]
                        perturb_forward["wings"][wing]["connect"]["dx"] = copy.copy(x1)
                        perturb_forward["wings"][wing]["connect"]["dy"] = copy.copy(y1)

                        if "v" in wing: # Vertical stabilizer
                            if perturb_forward["wings"][wing]["side"] == "right":
                                perturb_forward["wings"][wing]["mounting_angle"] += self.dtheta
                            else:
                                perturb_forward["wings"][wing]["mounting_angle"] -= self.dtheta
                        else: # Horizontal surface
                            if perturb_forward["wings"][wing]["side"] == "right":
                                perturb_forward["wings"][wing]["sweep"] -= self.dtheta
                            else:
                                perturb_forward["wings"][wing]["sweep"] += self.dtheta

        forward_file = wrt+"_forward.json"
        with open(forward_file, 'w', newline='\r\n') as dump_file:
            json.dump(perturb_forward,dump_file,indent=4)

        backward_file = wrt+"_backward.json"
        with open(backward_file, 'w', newline='\r\n') as dump_file:
            json.dump(perturb_backward,dump_file,indent=4)

        os.system("./Machup.out "+forward_file)
        os.system("./Machup.out "+backward_file)

        return forward_file, backward_file

    def get_forces_and_moments(self,filename):
        filename = filename.replace(".json","_forces.json")

        with open(filename, 'r') as coefs_file:
            coefs = json.load(coefs_file)

        q_inf = 0.5*self.combined_dict["condition"]["density"]*self.combined_dict["condition"]["velocity"]**2
        S_w = self.combined_dict["reference"]["area"]
        l_ref_lon = self.combined_dict["reference"]["longitudinal_length"]
        l_ref_lat = self.combined_dict["reference"]["lateral_length"]

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

        mom_trans = np.cross(self.r,tail_FM[0])
        tail_FM[1] -= mom_trans

        return lead_FM,tail_FM

    def calc_cent_diff(self,wrt):
        """Calculate a central difference approximation of the first derivative of forces and moments on the tailing aircraft with respect to a specified variable."""
        temp = self._run_perturbation(wrt)
        forward_file = temp[0]
        backward_file = temp[1]
        _,fwd_tail = self.get_forces_and_moments(forward_file)
        _,bwd_tail = self.get_forces_and_moments(backward_file)

        if wrt in "dxdydz":
            derivs = (fwd_tail-bwd_tail)/(2*self.dx)
        else:
            derivs = (fwd_tail-bwd_tail)/(2*self.dtheta)
        return derivs

    def run_derivs(self):
        """Find the matrix of derivatives of aerodynamic forces and moments with respect to position and orientation."""

        # d/dx
        dFM_dx = self.calc_cent_diff("dx")

        # d/dy
        dFM_dy = self.calc_cent_diff("dy")

        # d/dz
        dFM_dz = self.calc_cent_diff("dz")

        # d/dp
        dFM_dp = self.calc_cent_diff("dp")

        # d/dq
        dFM_dq = self.calc_cent_diff("dq")

        # d/dr
        dFM_dr = self.calc_cent_diff("dr")

        # d/dda
        dFM_dda = self.calc_cent_diff("aileron")*180/np.pi

        # d/dde
        dFM_dde = self.calc_cent_diff("elevator")*180/np.pi

        # d/ddr
        dFM_ddr = self.calc_cent_diff("rudder")*180/np.pi

        print("\nDerivatives w.r.t. x:\n{0}".format(dFM_dx))
        print("\nDerivatives w.r.t. y:\n{0}".format(dFM_dy))
        print("\nDerivatives w.r.t. z:\n{0}".format(dFM_dz))

        print("\nDerivatives w.r.t. p:\n{0}".format(dFM_dp))
        print("\nDerivatives w.r.t. q:\n{0}".format(dFM_dq))
        print("\nDerivatives w.r.t. r:\n{0}".format(dFM_dr))

        print("\nDerivatives w.r.t. da:\n{0}".format(dFM_dda))
        print("\nDerivatives w.r.t. de:\n{0}".format(dFM_dde))
        print("\nDerivatives w.r.t. dr:\n{0}".format(dFM_ddr))
        
        self.F_pbar = np.asarray([dFM_dx.flatten(),dFM_dy.flatten(),dFM_dz.flatten(),dFM_dp.flatten(),dFM_dq.flatten(),dFM_dr.flatten()])

        self.F_u = np.asarray([dFM_dda.flatten(),dFM_dde.flatten(),dFM_ddr.flatten()])

def grid_convergence(lead_file,tail_file,grids,separation_vec,trim_iterations=1):
    N = len(grids)
    state_derivs = []
    control_derivs = []

    # Run derivatives for each grid size
    for grid in grids:
        situ = TailingScenario(lead_file,tail_file,grid=grid)
        situ.apply_separation_and_trim(separation_vec,iterations=trim_iterations)
        situ.run_derivs()
        state_derivs.append(situ.F_pbar)
        control_derivs.append(situ.F_u)

    FM_names = ["Fx","Fy","Fz","Mx","My","Mz"]
    state_names = ["x","y","z","p","q","r"]

    state_derivs = np.asarray(state_derivs)

    # Create plot folder
    plot_dir = "./ConvergencePlots/"
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    # Plot convergence results
    shape = state_derivs[0].shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            plt.semilogx(grids,state_derivs[:,i,j],'x-')
            plt.xlabel("Grid Points")
            deriv_name = "d"+FM_names[j]+"/d"+state_names[i]
            plt.ylabel(deriv_name)
            plt.savefig(plot_dir+deriv_name.replace("/","_"))

if __name__=="__main__":

    # Initialize
    os.system("rm *forward*.json")
    os.system("rm *backward*.json")
    os.system("rm *trim*.json")

    # Define displacement of trailing aircraft
    r_CG = [-200,0,50]

    # Run scenario
    lead = "./IndividualModels/C130.json"
    tail = "./IndividualModels/ALTIUSjr.json"
    #situ = TailingScenario(lead,tail,grid=40)
    #situ.apply_separation_and_trim(r_CG,iterations=2)
    #situ.run_derivs()
    #print("\nFp:\n{0}".format(situ.F_pbar))
    #print("\nFu:\n{0}".format(situ.F_u))

    # Check grid convergence of derivatives
    grids = [10,20,50,100]
    grid_convergence(lead,tail,grids,r_CG)