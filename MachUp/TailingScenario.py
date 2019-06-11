import numpy as np
import json
import os
import copy
import scipy.optimize as opt
import matplotlib.pyplot as plt
import multiprocessing as mp
import subprocess as sp
import random
import time
import scipy.stats as stats

class TailingScenario:
    """Defines the situation of one aircraft trailing another. Separation is referenced from the CG of the leading aircraft in wind coordinates.
    Both aircraft are assumed to be aligned with the freestream, in yaw and roll at least, when trimmed. Due to limitations of numerical lifting-
    line, this analysis has been written to assume the tailing aircraft has no vertical stabilizer. The analysis will also not consider
    perturbations in yaw.
    
    Params:
    
    lead_filename (str):
        A .json file which is the MachUp input for the leading aircraft.
        
    tail_filename (str):
        A .json file which is the MachUp input for the tailing aircraft.
        
    dx (float, optional):
        Increment to be used in calculating derivatives with respect to position. Defaults to 0.01.

    dtheta (float, optional):
        Increment to be used in calculating derivatives with respect to orientation or control deflection.
        Defaults to 0.01.

    grid (int, optional):
        Number of control points to be used on each lifting surface in numerical lifting line. Defaults to 40.
    """

    def __init__(self,lead_filename,tail_filename,**kwargs):
        self.lead_filename = lead_filename
        self.tail_filename = tail_filename
        self.dx = kwargs.get("dx",0.01)
        self.dtheta = kwargs.get("dtheta",0.01)
        self.grid = kwargs.get("grid",40)

        self.FM_names = ["Fx","Fy","Fz","Mx","My","Mz"]
        self.state_names = ["x","y","z","phi","theta","psi"]
        self.control_names = ["aileron","elevator","rudder"]

        self._combine()

        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman"]+plt.rcParams["font.serif"]

    def _combine(self):
        """Combines two airplane .json files."""

        # Load initial files
        with open(self.lead_filename,'r') as lead_file_handle:
            self.lead_dict = json.load(lead_file_handle)

        with open(self.tail_filename,'r') as tail_file_handle:
            self.tail_dict = json.load(tail_file_handle)

        # Create new copy
        self.combined_dict = copy.deepcopy(self.lead_dict)

        # Alter some key values
        del self.combined_dict["solver"]["convergence"] # Use default convergence to avoid FORTRAN not being able to read Python-exported floating points
        self.combined_dict["solver"]["type"] = "nonlinear"
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
            if "v" in key: # Skip vertical stabilizers for now
                continue
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
            if "v" in key: # Skip vertical stabilizers for now
                continue
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

    def _apply_new_grid(self,grid):
        """Sets the number of grid points for each lifting surface."""
        self.grid = grid

        # Apply to base combined dict
        for wing in self.combined_dict["wings"]:
            self.combined_dict["wings"][wing]["grid"] = self.grid

        if hasattr(self,"trimmed_dict"):
            # Apply to trimmed dict
            for wing in self.trimmed_dict["wings"]:
                self.trimmed_dict["wings"][wing]["grid"] = self.grid
        
    def _apply_alpha_de(self,alpha,de,airplane):
        """Applies an angle of attack and elevator deflection to the specified airplane."""

        # Apply elevator deflection
        self.trimmed_dict["controls"][airplane+"_elevator"]["deflection"] = de

        # Apply alpha
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

        # Store alpha
        try:
            self.trimmed_dict["angles_of_attack"][airplane] = alpha
        except KeyError:
            self.trimmed_dict["angles_of_attack"] = {}
            self.trimmed_dict["angles_of_attack"][airplane] = alpha
        
        # Save output
        with open(self.trimmed_filename,'w') as dump_file_handle:
            json.dump(self.trimmed_dict,dump_file_handle,indent=4)

    def _run_machup(self,input_filename):
        try:
            print("Running {0}".format(input_filename))
            with open(os.devnull,'w') as FNULL:
                completed = sp.run(["./Machup.out",input_filename],stdout=FNULL,stderr=sp.STDOUT)
            if completed.returncode < 0:
                print("MachUp terminated with code {0} after trying to execute {1}".format(-completed.returncode,input_filename))
            else:
                print("MachUp successfully completed executing {0}".format(input_filename))
        except OSError as e:
            raise RuntimeError("MachUp execution failed while executing {0} due to {1}".format(input_filename,e))

    def _get_trim_residuals(self,alpha,de,airplane):
        """Returns L-W and m for the given aircraft in the given state."""

        # Apply angle of attack and elevator deflection and run
        self._apply_alpha_de(alpha,de,airplane)
        self._run_machup(self.trimmed_filename)
        lead_FM,tail_FM = self._get_forces_and_moments(self.trimmed_filename)

        # Calculate residuals
        if airplane == "tail":
            R_L = np.asscalar(-tail_FM[0,2])-self.combined_dict["condition"]["W_"+airplane]
            R_m = np.asscalar(tail_FM[1,1])
        else:
            R_L = np.asscalar(-lead_FM[0,2])-self.combined_dict["condition"]["W_"+airplane]
            R_m = np.asscalar(lead_FM[1,1])

        return np.asarray([R_L,R_m])

    def _trim(self,alpha_init,de_init,airplane,convergence):
        """Uses Newton's method to solve for trim."""
        alpha0 = copy.deepcopy(alpha_init)
        de0 = copy.deepcopy(de_init)

        # Calculate derivatives and residuals
        # This doesn't use self._calc_central_diff() in order to run faster with high grid resolutions
        arg_list = [(airplane,{"dtheta":self.dtheta}),(airplane,{"dtheta":-self.dtheta}),(airplane,{"elevator":self.dtheta}),(airplane,{"elevator":-self.dtheta})]
        with mp.Pool() as pool:
            R_get = pool.apply_async(self._get_trim_residuals,(alpha_init,de_init,airplane))
            FM_get = pool.map_async(self._get_perturbed_forces_and_moments,arg_list)
            FM = FM_get.get()
            R = R_get.get()

        h = 2*self.dtheta
        derivs = []
        derivs.append((FM[0]-FM[1])/h)
        derivs.append((FM[2]-FM[3])/h)

        # Output progress
        print("Trim residuals:\n    Fz: {0}\n    My: {1}".format(R[0],R[1]))
    
        while (abs(R)>convergence).any(): # If the trim conditions have not yet converged

            # Arrange Jacobian
            J = np.zeros((2,2))
            J[0,0] = -derivs[0][0,2] # dL/da
            J[0,1] = -derivs[1][0,2] # dL/dde
            J[1,0] = derivs[0][1,1] # dm/da
            J[1,1] = derivs[1][1,1] # dm/dde

            # Correct pitch and elevator deflection
            corrector = np.asarray(np.linalg.inv(J)*np.asmatrix(R).T)
            alpha1 = np.asscalar(alpha0-corrector[0])
            de1 = np.asscalar(de0-corrector[1])

            # Calculate derivatives and residuals
            arg_list = [(airplane,{"dtheta":self.dtheta}),(airplane,{"dtheta":-self.dtheta}),(airplane,{"elevator":self.dtheta}),(airplane,{"elevator":-self.dtheta})]
            with mp.Pool() as pool:
                R_get = pool.apply_async(self._get_trim_residuals,(alpha1,de1,airplane))
                FM_get = pool.map_async(self._get_perturbed_forces_and_moments,arg_list)
                FM = FM_get.get()
                R = R_get.get()

            h = 2*self.dtheta
            derivs = []
            derivs.append((FM[0]-FM[1])/h)
            derivs.append((FM[2]-FM[3])/h)

            # Output progress
            print("Trim residuals:\n    Fz: {0}\n    My: {1}".format(R[0],R[1]))

            # Update for next iteration
            alpha0 = alpha1
            de0 = de1
            sp.run(["rm","*perturbed*"])

        self._apply_alpha_de(alpha0,de0,airplane) # Because this was done in another process

    def _apply_separation_and_trim(self,separation_vec,iterations=1,grid=40,convergence=1e-8,export_stl=False):
        """Separates the two aircraft according to separation_vec and trims both.
        This will leave alpha and beta as 0.0, instead trimming using mounting
        angles. This will first trim the leading aircraft, then the tailing aircraft.
        This process can be repeated for finer trim results using the iterations
        parameter."""

        self._apply_new_grid(grid)

        # If the trimmed dict does not already exist, create it
        # This is done so that subsequent trimming operations have a better starting point.
        if not hasattr(self,"trimmed_dict"):
            self.trimmed_dict = copy.deepcopy(self.combined_dict)

        # Apply separation
        self.r = separation_vec
        for wing in self.trimmed_dict["wings"]:
            if "tail" in wing:
                self.trimmed_dict["wings"][wing]["connect"]["dx"] = self.combined_dict["wings"][wing]["connect"]["dx"]+self.r[0]
                self.trimmed_dict["wings"][wing]["connect"]["dy"] = self.combined_dict["wings"][wing]["connect"]["dy"]+self.r[1]
                self.trimmed_dict["wings"][wing]["connect"]["dz"] = self.combined_dict["wings"][wing]["connect"]["dz"]+self.r[2]
        
        # Dump info
        self.trimmed_filename = "trimmed.json"
        with open(self.trimmed_filename,'w') as dump_file_handle:
            json.dump(self.trimmed_dict,dump_file_handle,indent=4)

        # Trim each plane iteratively
        for _ in range(iterations):
            airplanes = ["lead","tail"]
            for airplane in airplanes:

                # Get an initial guess for the angle of attack and elevator deflection
                try:
                    alpha_init = self.trimmed_dict["angles_of_attack"][airplane]
                except KeyError:
                    if airplane == "lead":
                        alpha_init = self.lead_dict["condition"]["alpha"]
                    else:
                        alpha_init = self.tail_dict["condition"]["alpha"]

                de_init = self.trimmed_dict["controls"][airplane+"_elevator"]["deflection"]

                # Trim
                self._trim(alpha_init,de_init,airplane,convergence)

        # Export .stl file of the trimmed situation if specified
        if export_stl:
            stl_dict = copy.deepcopy(self.trimmed_dict)
            stl_dict["run"] = {"stl": ""}

            run_stl_filename = "trimmed_stl.json"
            with open(run_stl_filename,'w') as stl_file_handle:
                json.dump(stl_dict,stl_file_handle,indent=4)

            self._run_machup(run_stl_filename)
            sp.run(["rm",run_stl_filename])

    def _perturb_position(self,airplane_dict,wrt,perturbation,airplane):
        """Perturbs the position of the aircraft in the direction of wrt."""

        # Move each lifting surface according to the perturbation
        for key in airplane_dict["wings"]:
            if airplane in key:
                airplane_dict["wings"][key]["connect"][wrt] += perturbation

        return airplane_dict

    def _perturb_control(self,airplane_dict,wrt,perturbation,airplane):
        """Perturbs the specified control of the aircraft."""

        # Move the control according to the perturbation
        control_key = airplane+"_"+wrt
        airplane_dict["controls"][control_key]["deflection"] += perturbation

        return airplane_dict

    def _perturb_pitch(self,airplane_dict,perturbation,airplane):
        """Perturbs the specified aircraft in pitch."""
        for wing in airplane_dict["wings"]:
            if airplane in wing:

                # Apply to position
                x0 = copy.deepcopy(airplane_dict["wings"][wing]["connect"]["dx"])
                z0 = copy.deepcopy(airplane_dict["wings"][wing]["connect"]["dz"])
                if airplane == "tail":
                    x0 -= self.r[0]
                    z0 -= self.r[2]
                C = np.cos(np.radians(perturbation))
                S = np.sin(np.radians(perturbation))
                x1 = x0*C+z0*S
                z1 = z0*C-x0*S
                if airplane == "tail":
                    x1 += self.r[0]
                    z1 += self.r[2]
                airplane_dict["wings"][wing]["connect"]["dx"] = copy.copy(x1)
                airplane_dict["wings"][wing]["connect"]["dz"] = copy.copy(z1)

                # Apply to angle
                if "v" in wing: # Vertical stabilizer
                    airplane_dict["wings"][wing]["sweep"] += perturbation
                else: # Horizontal surface
                    airplane_dict["wings"][wing]["mounting_angle"] += perturbation

        return airplane_dict

    def _perturb_roll(self,airplane_dict,perturbation,airplane):
        """Perturbs the specified aircraft in roll."""
        for wing in airplane_dict["wings"]:
            if airplane in wing:

                # Apply to position
                y0 = copy.deepcopy(airplane_dict["wings"][wing]["connect"]["dy"])
                z0 = copy.deepcopy(airplane_dict["wings"][wing]["connect"]["dz"])
                if airplane == "tail":
                    y0 -= self.r[1]
                    z0 -= self.r[2]
                C = np.cos(np.radians(perturbation))
                S = np.sin(np.radians(perturbation))
                y1 = y0*C-z0*S
                z1 = z0*C+y0*S
                if airplane == "tail":
                    y1 += self.r[1]
                    z1 += self.r[2]
                airplane_dict["wings"][wing]["connect"]["dy"] = copy.copy(y1)
                airplane_dict["wings"][wing]["connect"]["dz"] = copy.copy(z1)

                # Apply to angle
                if airplane_dict["wings"][wing]["side"] == "right":
                    airplane_dict["wings"][wing]["dihedral"] -= perturbation
                else:
                    airplane_dict["wings"][wing]["dihedral"] += perturbation

        return airplane_dict

    def _perturb_yaw(self,airplane_dict,perturbation,airplane):
        """Perturbs the specified aircraft in yaw."""
        for wing in airplane_dict["wings"]:
            if airplane in wing:

                # Apply to position
                x0 = copy.deepcopy(airplane_dict["wings"][wing]["connect"]["dx"])
                y0 = copy.deepcopy(airplane_dict["wings"][wing]["connect"]["dy"])
                if airplane == "tail":
                    x0 -= self.r[0]
                    y0 -= self.r[1]
                C = np.cos(np.radians(self.dtheta))
                S = np.sin(np.radians(self.dtheta))
                x1 = x0*C-y0*S
                y1 = y0*C+x0*S
                if airplane == "tail":
                    x1 += self.r[0]
                    y1 += self.r[1]
                airplane_dict["wings"][wing]["connect"]["dx"] = copy.copy(x1)
                airplane_dict["wings"][wing]["connect"]["dy"] = copy.copy(y1)

                # Apply to angle
                if "v" in wing: # Vertical stabilizer
                    if airplane_dict["wings"][wing]["side"] == "right":
                        airplane_dict["wings"][wing]["mounting_angle"] -= perturbation
                    else:
                        airplane_dict["wings"][wing]["mounting_angle"] += perturbation
                else: # Horizontal surface
                    if airplane_dict["wings"][wing]["side"] == "right":
                        airplane_dict["wings"][wing]["sweep"] += perturbation
                    else:
                        airplane_dict["wings"][wing]["sweep"] -= perturbation

        return airplane_dict

    def _run_perturbation(self,airplane,perturbations):
        """Perturbs the specified aircraft according to the perturbations dict."""
        perturbed = copy.deepcopy(self.trimmed_dict)

        # Create descriptive file tag
        tag_list = []
        for key in perturbations:
            tag_list.append(key+str(perturbations[key]))
        tag = "".join(tag_list)

        # Apply specified perturbations
        for key in perturbations: # The key specifies the variable to perturb

            if key in "dxdydz": # Position
                perturbed = self._perturb_position(perturbed,key,perturbations[key],airplane)

            elif key in "aileronelevatorrudder": # Control input
                perturbed = self._perturb_control(perturbed,key,perturbations[key],airplane)

            elif key == "dphi": # Roll
                perturbed = self._perturb_roll(perturbed,perturbations[key],airplane)
        
            elif key == "dtheta": # Pitch
                perturbed = self._perturb_pitch(perturbed,perturbations[key],airplane)

            elif key == "dpsi": # Yaw
                perturbed = self._perturb_yaw(perturbed,perturbations[key],airplane)

        # Store and run perturbed dict
        perturbed_file = tag+"perturbed.json"
        with open(perturbed_file, 'w', newline='\r\n') as dump_file:
            json.dump(perturbed,dump_file,indent=4)

        self._run_machup(perturbed_file)

        return perturbed_file

    def _get_forces_and_moments(self,filename):
        """Extracts aerodynamic forces and moments from the specified MachUp output file."""

        # Define reference constants
        q_inf = 0.5*self.combined_dict["condition"]["density"]*self.combined_dict["condition"]["velocity"]**2
        S_w = self.combined_dict["reference"]["area"]
        l_ref_lon = self.combined_dict["reference"]["longitudinal_length"]
        l_ref_lat = self.combined_dict["reference"]["lateral_length"]

        # Load force and moment coefficients
        filename = filename.replace(".json","_forces.json")
        with open(filename, 'r') as coefs_file:
            coefs = json.load(coefs_file)

        lead_FM = np.zeros((2,3))
        tail_FM = np.zeros((2,3))

        # Dimensionalize and sum across all lifting surfaces
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

        # For tail lifting surfaces, transform the moments to be about the tailing CG
        mom_trans = np.cross(self.r,tail_FM[0])
        tail_FM[1] -= mom_trans

        return lead_FM,tail_FM

    def _get_perturbed_forces_and_moments(self,args):
        """Perturbs the specified aircraft from trim and extracts the forces and moments."""
        airplane,perturbations = args

        # Perturb and extract forces and moments
        filename = self._run_perturbation(airplane,perturbations)
        lead_FM,tail_FM = self._get_forces_and_moments(filename)

        # Clean up Machup files
        sp.run(["rm",filename])
        sp.run(["rm",filename.replace(".json","_forces.json")])

        if airplane == "lead":
            return lead_FM
        else:
            return tail_FM

    def _calc_cent_diff(self,args):
        """Calculate a central difference approximation of the first derivative of forces and moments on the specified aircraft with respect to a specified variable."""
        wrt,airplane = args
        if wrt in "dxdydz": # Linear
            fwd_FM = self._get_perturbed_forces_and_moments((airplane,{wrt:self.dx}))
            bwd_FM = self._get_perturbed_forces_and_moments((airplane,{wrt:-self.dx}))
            h = 2*self.dx

        else: # Angular
            fwd_FM = self._get_perturbed_forces_and_moments((airplane,{wrt:self.dtheta}))
            bwd_FM = self._get_perturbed_forces_and_moments((airplane,{wrt:-self.dtheta}))
            h = 2*self.dtheta

        derivs = (fwd_FM-bwd_FM)/h

        return derivs

    def run_derivs(self,separation_vec,trim_iterations=2,grid=40,export_stl=False):
        """Find the matrix of derivatives of aerodynamic forces and moments with respect to position, orientation, and control input."""
        self._apply_separation_and_trim(separation_vec,iterations=trim_iterations,grid=grid,export_stl=export_stl)

        # Distribute calculations among processes
        deriv_list = []
        for state_var in self.state_names:
            deriv_list.append("d"+state_var)
        for control in self.control_names:
            deriv_list.append(control)

        arg_list = [(deriv,"tail") for deriv in deriv_list]
        with mp.Pool() as pool:
            F_pbar_u = pool.map(self._calc_cent_diff,arg_list)

        # Format derivative matrices
        for i in range(len(F_pbar_u)):
            F_pbar_u[i] = F_pbar_u[i].flatten()

        self.F_pbar = np.asarray(F_pbar_u[:6])
        self.F_u = np.asarray(F_pbar_u[6:])

    def plot_derivative_convergence(self,grids,separation_vec,trim_iterations=2):
        state_derivs = []
        control_derivs = []

        # Run derivatives for each grid size
        for grid in grids:
            situ.run_derivs(separation_vec,trim_iterations=trim_iterations,grid=grid)
            state_derivs.append(situ.F_pbar)
            control_derivs.append(situ.F_u)

        state_derivs = np.asarray(state_derivs)
        control_derivs = np.asarray(control_derivs)

        # Create plot folder
        plot_dir = "./ConvergencePlots/"
        if not os.path.exists(plot_dir):
            os.mkdir(plot_dir)

        # Plot convergence results
        # State derivatives
        shape = state_derivs[0].shape
        for i in range(shape[0]):
            for j in range(shape[1]):
                plt.figure()
                plt.semilogx(grids,state_derivs[:,i,j],'kx--')
                plt.xlabel("Grid Points")
                deriv_name = "d"+self.FM_names[j]+"/d"+self.state_names[i]
                plt.ylabel(deriv_name)
                plt.savefig(plot_dir+deriv_name.replace("/","_"),bbox_inches='tight')
                plt.close()
                
        # Control derivatives
        shape = control_derivs[0].shape
        for i in range(shape[0]):
            for j in range(shape[1]):
                plt.figure()
                plt.semilogx(grids,control_derivs[:,i,j],'kx--')
                plt.xlabel("Grid Points")
                deriv_name = "d"+self.FM_names[j]+"/dd"+self.control_names[i][0]
                plt.ylabel(deriv_name)
                plt.savefig(plot_dir+deriv_name.replace("/","_"),bbox_inches='tight')
                plt.close()

    def _get_normal_perturbations(self,dispersions,sigma):
        """Generates a dict of normally distributed perturbations in the variables specified by dispersions."""
        perturbations = {}

        # Get normally distributed variable for each perturbation
        for key in dispersions:
            perturbations[key] = np.random.normal(0.0,dispersions[key]/sigma)

        return perturbations

    def run_monte_carlo(self,separation_vec,airplane,N_samples,dispersions,grid=40,sigma=3):
        """Runs a Monte Carlo simulation to determine the dispersion of forces and moments."""
        self._apply_separation_and_trim(separation_vec,iterations=2,grid=grid)

        # Apply to multiprocessing
        start_time = time.time()
        arg_list = [(airplane,self._get_normal_perturbations(dispersions,sigma)) for i in range(N_samples)]
        with mp.Pool() as pool:
            FM_samples = pool.map(self._get_perturbed_forces_and_moments,arg_list)

        # Determine force and moment dispersions
        FM_samples = np.asarray(FM_samples)
        self.FM_dispersions = np.std(FM_samples,axis=0)
        end_time = time.time()

        # Plot dispersions
        plot_dir = "./MCDispersions"
        if not os.path.exists(plot_dir):
            os.mkdir(plot_dir)
        
        shape = FM_samples[0].shape
        self.MC_dispersions = np.zeros(shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                plt.figure()
                name = self.FM_names[i*shape[1]+j]
                
                # Plot histogram
                x = FM_samples[:,i,j].flatten()
                plt.hist(x,density=True,label=name)

                # Plot normal distribution
                x_space = np.linspace(min(x),max(x),100)
                x_mean_info,_,x_std_info = stats.bayes_mvs(x,alpha=0.95)
                x_mean = x_mean_info.statistic
                x_std = x_std_info.statistic
                self.MC_dispersions[i,j] = x_std
                fit = stats.norm.pdf(x_space,x_mean,x_std)
                plt.plot(x_space,fit,"k-",label="Normal PDF")

                # Plot confidence interval of standard deviation
                x_std_upper = x_std_info.minmax[1]
                x_std_lower = x_std_info.minmax[0]
                upper_fit = stats.norm.pdf(x_space,x_mean,x_std_upper)
                lower_fit = stats.norm.pdf(x_space,x_mean,x_std_lower)
                plt.plot(x_space,upper_fit,"k--")
                plt.plot(x_space,lower_fit,"k--")

                # Format and save
                plt.xlabel(name)
                plt.legend()
                plt.savefig(plot_dir+"/"+name)
                plt.close()

        return end_time-start_time

    def run_lin_cov(self,separation_vec,airplane,dispersions,trim_iterations=2,grid=40):
        """Runs linear covariance analysis on the specified airplane using the given dispersions."""

        # Calculate derivatives
        self.run_derivs(separation_vec,trim_iterations=trim_iterations,grid=grid)

    def plot_perturbation_convergence(self,separation_vec,airplane,perturbation,grids,trim_iterations=2,trim_once=True):
        """Plots the grid convergence of the forces and moments generated by a given perturbation."""

        # Run perturbation at each grid level
        FM_list = []
        alpha_lead_list = []
        alpha_tail_list = []

        # If only trimming once
        if trim_once:
            self._apply_separation_and_trim(separation_vec,iterations=trim_iterations)
        for grid in grids:
            self._apply_new_grid(grid)
            
            # If trimming at each grid level
            if not trim_once:
                self._apply_separation_and_trim(separation_vec,iterations=trim_iterations,grid=grid)

            FM_list.append(self._get_perturbed_forces_and_moments((airplane,perturbation)))
            alpha_lead_list.append(self.trimmed_dict["angles_of_attack"]["lead"])
            alpha_tail_list.append(self.trimmed_dict["angles_of_attack"]["tail"])

        # Plot
        var = list(perturbation.keys())[0]
        if trim_once:
            plot_dir = "./ForceAndMomentConvergencePlotsTrimmedOnceWRT"+var
        else:
            plot_dir = "./ForceAndMomentConvergencePlotsTrimmedAtEachWRT"+var

        if not os.path.exists(plot_dir):
            os.mkdir(plot_dir)

        FM = np.asarray(FM_list)
        shape = FM[0].shape
        for i in range(shape[0]):
            for j in range(shape[1]):
                plt.figure()
                plt.semilogx(grids,FM[:,i,j],"kx--")
                name = self.FM_names[i*shape[1]+j]
                plt.xlabel("Grid Points")
                plt.ylabel(name)
                plt.savefig(plot_dir+"/"+name)

        # Angles of attack
        plt.figure()
        plt.plot(grids,alpha_lead_list,"kx--")
        plt.xlabel("Grid Points")
        plt.ylabel("Lead Angle of Attack")
        plt.savefig(plot_dir+"/LeadAlpha")

        plt.figure()
        plt.plot(grids,alpha_tail_list,"kx--")
        plt.xlabel("Grid Points")
        plt.ylabel("Tail Angle of Attack")
        plt.savefig(plot_dir+"/TailAlpha")

if __name__=="__main__":

    # Initialize
    sp.run(["rm","*forward*"])
    sp.run(["rm","*backward*"])
    sp.run(["rm","*perturbed*"])
    sp.run(["rm","*trim*"])

    # Define displacement of trailing aircraft
    r_CG = [-200,0,50]

    # Initialize scenario
    lead = "./IndividualModels/C130.json"
    tail = "./IndividualModels/ALTIUSjr.json"
    situ = TailingScenario(lead,tail)

    # Run sensitivities
    #situ.run_derivs(r_CG,export_stl=True)
    #print("\nFp:\n{0}".format(situ.F_pbar))
    #print("\nFu:\n{0}".format(situ.F_u))

    # Check force and moment grid convergence
    grid_floats = np.logspace(1,2.5,10)
    grids = list(map(int,grid_floats))
    perturb = {
        "dtheta": 0.1
    }
    situ.plot_perturbation_convergence(r_CG,"tail",perturb,grids)
    situ.plot_perturbation_convergence(r_CG,"tail",perturb,grids,trim_once=False)

    perturb = {
        "dphi": 0.1
    }
    situ.plot_perturbation_convergence(r_CG,"tail",perturb,grids)
    situ.plot_perturbation_convergence(r_CG,"tail",perturb,grids,trim_once=False)

    perturb = {
        "dx": 0.1
    }
    situ.plot_perturbation_convergence(r_CG,"tail",perturb,grids)
    situ.plot_perturbation_convergence(r_CG,"tail",perturb,grids,trim_once=False)

    perturb = {
        "dy": 0.1
    }
    situ.plot_perturbation_convergence(r_CG,"tail",perturb,grids)
    situ.plot_perturbation_convergence(r_CG,"tail",perturb,grids,trim_once=False)

    perturb = {
        "dz": 0.1
    }
    situ.plot_perturbation_convergence(r_CG,"tail",perturb,grids)
    situ.plot_perturbation_convergence(r_CG,"tail",perturb,grids,trim_once=False)

    # Check grid convergence of derivatives
    situ.plot_derivative_convergence(grids,r_CG)

    # Run Monte Carlo simulation
    dispersions = {
        "dx": 5.0,
        "dy": 5.0,
        "dz": 5.0,
        "dphi": 10*np.pi/180,
        "dtheta": 10*np.pi/180,
        "aileron": 5*np.pi/180,
        "elevator": 5*np.pi/180
    }
    #MC_exec_time = situ.run_monte_carlo(r_CG,"tail",1000,dispersions)
    #print("Monte Carlo took {0} s to run.".format(MC_exec_time))
    #print(situ.FM_dispersions)

    # Run LinCov
    #situ.run_lin_cov(r_CG,"tail",dispersions)