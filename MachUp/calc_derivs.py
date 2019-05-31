import numpy as np
import json
import os
import copy
import scipy.optimize as opt

class TailingScenario:
    """Defines the situation of one aircraft trailing another. Separation is referenced from the leading aircraft in aircraft coordinates."""

    def __init__(self,lead_filename,tail_filename,**kwargs):
        self.lead_filename = lead_filename
        self.tail_filename = tail_filename
        self.dx = kwargs.get("dx",0.1)
        self.dtheta = kwargs.get("dtheta",0.1)

        self.combine()

    def combine(self):
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

        del self.combined_dict["condition"]["W"]
        self.combined_dict["condition"]["W_lead"] = self.lead_dict["condition"]["W"]
        self.combined_dict["condition"]["W_tail"] = self.tail_dict["condition"]["W"]

        for key in self.lead_dict["controls"]:
            new_key = "lead_"+key
            self.combined_dict["controls"][new_key] = self.lead_dict["controls"][key]

        for key in self.lead_dict["wings"]:
            new_key = "lead_"+key
            self.combined_dict["wings"][new_key] = self.lead_dict["wings"][key]
            for control_key in self.combined_dict["wings"][new_key]["control"]["mix"]:
                self.combined_dict["wings"][new_key]["control"]["mix"] = {"lead_"+control_key:1.0}

        for key in self.tail_dict["controls"]:
            new_key = "tail_"+key
            self.combined_dict["controls"][new_key] = self.tail_dict["controls"][key]

        for key in self.tail_dict["wings"]:
            new_key = "tail_"+key
            self.combined_dict["wings"][new_key] = self.tail_dict["wings"][key]
            for control_key in self.combined_dict["wings"][new_key]["control"]["mix"]:
                self.combined_dict["wings"][new_key]["control"]["mix"] = {"tail_"+control_key:1.0}

        # Create new .json file
        self.untrimmed_filename = "combined_untrimmed.json"
        with open(self.untrimmed_filename,'w') as dump_file_handle:
            json.dump(self.combined_dict,dump_file_handle,indent=4)
        
    def apply_alpha_de(self,alpha,de,airplane):
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

    def get_trim_residuals(self,x,airplane):
        """Returns L-W and m for the given aircraft in the given state."""
        alpha,de = x
        self.apply_alpha_de(alpha,de,airplane)
        os.system("./Machup.out "+self.trimmed_filename)
        lead_FM,tail_FM = self.get_forces_and_moments(self.trimmed_filename)
        if airplane == "tail":
            R_L = np.asscalar(-tail_FM[0,2])-self.combined_dict["condition"]["W_"+airplane]
            R_m = np.asscalar(tail_FM[1,1])
        else:
            R_L = np.asscalar(-lead_FM[0,2])-self.combined_dict["condition"]["W_"+airplane]
            R_m = np.asscalar(lead_FM[1,1])
        return R_L,R_m

    def apply_separation_and_trim(self,separation_vec):
        """Separates the two aircraft according to separation_vec and trims both.
        This will leave alpha and beta as 0.0, instead trimming using mounting
        angles."""
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

        airplanes = ["lead","tail"]
        for airplane in airplanes:
            if airplane == "lead":
                alpha_init = self.lead_dict["condition"]["alpha"]
            else:
                alpha_init = self.tail_dict["condition"]["alpha"]

            de_init = self.trimmed_dict["controls"][airplane+"_elevator"]["deflection"]

            _ = opt.root(self.get_trim_residuals,[alpha_init,de_init],args=(airplane,))

    def run_perturbation(self,wrt):
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
            for key in self.trimmed_dict["wings"]:
                if "tail" in key:
                    pass

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
        print("q_inf: {0}".format(q_inf))
        print("Sw: {0}".format(S_w))
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
        """Calculate a central difference approximation of the first derivative of forces and moments with respect to a specified variable."""
        temp = self.run_perturbation(wrt)
        forward_file = temp[0]
        backward_file = temp[1]
        fwd_lead,fwd_tail = self.get_forces_and_moments(forward_file)
        bwd_lead,bwd_tail = self.get_forces_and_moments(backward_file)

        if wrt in "dxdydz":
            derivs = (fwd_tail-bwd_tail)/(2*self.dx)
        else:
            derivs = (fwd_tail-bwd_tail)/(2*self.dtheta)
        return derivs

    def run(self):
        """Find the matrix of derivatives of aerodynamic forces and moments with respect to position and orientation."""

        # d/dx
        dFM_dx = self.calc_cent_diff("dx")

        # d/dy
        dFM_dy = self.calc_cent_diff("dy")

        # d/dz
        dFM_dz = self.calc_cent_diff("dz")

        # d/dda
        dFM_dda = self.calc_cent_diff("aileron")*180/np.pi

        # d/dde
        dFM_dde = self.calc_cent_diff("elevator")*180/np.pi

        # d/ddr
        dFM_ddr = self.calc_cent_diff("rudder")*180/np.pi

        print("\nDerivatives w.r.t. x:\n{0}".format(dFM_dx))
        print("\nDerivatives w.r.t. y:\n{0}".format(dFM_dy))
        print("\nDerivatives w.r.t. z:\n{0}".format(dFM_dz))

        print("\nDerivatives w.r.t. da:\n{0}".format(dFM_dda))
        print("\nDerivatives w.r.t. de:\n{0}".format(dFM_dde))
        print("\nDerivatives w.r.t. dr:\n{0}".format(dFM_ddr))
        
        self.F_pbar = np.asarray([dFM_dx.flatten(),dFM_dy.flatten(),dFM_dz.flatten()])
        print("\nFp:\n{0}".format(self.F_pbar))

        self.F_u = np.asarray([dFM_dda.flatten(),dFM_dde.flatten(),dFM_ddr.flatten()])
        print("\nFu:\n{0}".format(self.F_u))

if __name__=="__main__":

    # Initialize
    os.system("rm *forward*.json")
    os.system("rm *backward*.json")
    os.system("rm *trim*.json")

    # Define displacement of trailing aircraft
    r_CG = np.array([-200,0,50])

    # Run scenario
    situ = TailingScenario("./IndividualModels/C130.json","./IndividualModels/ALTIUSjr.json")
    situ.apply_separation_and_trim([-200.0,0.0,50.0])
    situ.run()