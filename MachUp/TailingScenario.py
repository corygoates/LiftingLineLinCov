import numpy as np
import json
import os
import copy
import scipy.optimize as opt
import matplotlib.pyplot as plt
import multiprocessing as mp
import random
import time
import scipy.stats as stats

class TailingScenario:
    """Defines the situation of one aircraft trailing another. Separation is referenced from the leading aircraft in aircraft coordinates."""

    def __init__(self,lead_filename,tail_filename,**kwargs):
        self.lead_filename = lead_filename
        self.tail_filename = tail_filename
        self.dx = kwargs.get("dx",0.01)
        self.dtheta = kwargs.get("dtheta",0.01)
        self.grid = kwargs.get("grid",40)

        self.FM_names = ["Fx","Fy","Fz","Mx","My","Mz"]
        self.state_names = ["x","y","z","p","q","r"]
        self.control_names = ["a","e","r"]

        self._combine()

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

    def _get_trim_residuals(self,alpha,de,airplane):
        """Returns L-W and m for the given aircraft in the given state."""
        self._apply_alpha_de(alpha,de,airplane)
        os.system("./Machup.out "+self.trimmed_filename)
        lead_FM,tail_FM = self.get_forces_and_moments(self.trimmed_filename)
        if airplane == "tail":
            R_L = np.asscalar(-tail_FM[0,2])-self.combined_dict["condition"]["W_"+airplane]
            R_m = np.asscalar(tail_FM[1,1])
        else:
            R_L = np.asscalar(-lead_FM[0,2])-self.combined_dict["condition"]["W_"+airplane]
            R_m = np.asscalar(lead_FM[1,1])
        return np.asarray([R_L,R_m])

    def _trim(self,alpha_init,de_init,airplane,convergence):
        """Uses Newton's method to solve for trim."""
        R = self._get_trim_residuals(alpha_init,de_init,airplane)
        alpha0 = copy.deepcopy(alpha_init)
        de0 = copy.deepcopy(de_init)

        while (abs(R)>convergence).any(): # If the trim conditions have not yet converged

            # Calculate derivatives
            arg_list = [("dq",airplane),("elevator",airplane)]
            with mp.Pool(mp.cpu_count()) as pool:
                derivs = pool.map(self.calc_cent_diff,arg_list)

            J = np.zeros((2,2))
            J[0,0] = -derivs[0][0,2] # dL/da
            J[0,1] = -derivs[1][0,2] # dL/dde
            J[1,0] = derivs[0][1,1] # dm/da
            J[1,1] = derivs[1][1,1] # dm/dde

            # Correct pitch and elevator deflection
            corrector = np.asarray(np.linalg.inv(J)*np.asmatrix(R).T)
            alpha1 = np.asscalar(alpha0-corrector[0])
            de1 = np.asscalar(de0-corrector[1])

            # Run corrected attitude
            R = self._get_trim_residuals(alpha1,de1,airplane)
            alpha0 = alpha1
            de0 = de1

    def apply_separation_and_trim(self,separation_vec,iterations=1,convergence=1e-8):
        """Separates the two aircraft according to separation_vec and trims both.
        This will leave alpha and beta as 0.0, instead trimming using mounting
        angles. This will first trim the leading aircraft, then the tailing aircraft.
        This process can be repeated for finer trim results using the iterations
        parameter."""

        # Apply separation
        self.r = separation_vec
        self.trimmed_dict = copy.deepcopy(self.combined_dict)
        for wing in self.trimmed_dict["wings"]:
            if "tail" in wing:
                self.trimmed_dict["wings"][wing]["connect"]["dx"] += self.r[0]
                self.trimmed_dict["wings"][wing]["connect"]["dy"] += self.r[1]
                self.trimmed_dict["wings"][wing]["connect"]["dz"] += self.r[2]
        
        # Dump info
        self.trimmed_filename = "trimmed.json"
        with open(self.trimmed_filename,'w') as dump_file_handle:
            json.dump(self.trimmed_dict,dump_file_handle,indent=4)

        # Trim each plane iteratively
        for _ in range(iterations):
            airplanes = ["lead","tail"]
            for airplane in airplanes:
                if airplane == "lead":
                    alpha_init = self.lead_dict["condition"]["alpha"]
                else:
                    alpha_init = self.tail_dict["condition"]["alpha"]

                de_init = self.trimmed_dict["controls"][airplane+"_elevator"]["deflection"]

                self._trim(alpha_init,de_init,airplane,convergence)
                #_ = opt.root(self._get_trim_residuals,[alpha_init,de_init],args=(airplane,))

    def _perturb_position(self,airplane_dict,wrt,perturbation,airplane):
        """Perturbs the position of the aircraft in the direction of wrt."""
        for key in airplane_dict["wings"]:
            if airplane in key:
                airplane_dict["wings"][key]["connect"][wrt] += perturbation

        return airplane_dict

    def _perturb_control(self,airplane_dict,wrt,perturbation,airplane):
        """Perturbs the specified control of the aircraft."""
        control_key = airplane+"_"+wrt
        airplane_dict["controls"][control_key]["deflection"] += perturbation

        return airplane_dict

    def _perturb_pitch(self,airplane_dict,perturbation,airplane):
        """Perturbs the specified aircraft in pitch."""
        for wing in airplane_dict["wings"]:
            if airplane in wing:
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

                if "v" in wing: # Vertical stabilizer
                    airplane_dict["wings"][wing]["sweep"] += perturbation
                else: # Horizontal surface
                    airplane_dict["wings"][wing]["mounting_angle"] += perturbation

        return airplane_dict

    def _perturb_roll(self,airplane_dict,perturbation,airplane):
        """Perturbs the specified aircraft in roll."""
        for wing in airplane_dict["wings"]:
            if airplane in wing:
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

                if airplane_dict["wings"][wing]["side"] == "right":
                    airplane_dict["wings"][wing]["dihedral"] -= perturbation
                else:
                    airplane_dict["wings"][wing]["dihedral"] += perturbation

        return airplane_dict

    def _perturb_yaw(self,airplane_dict,perturbation,airplane):
        """Perturbs the specified aircraft in yaw."""
        for wing in airplane_dict["wings"]:
            if airplane in wing:
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

    def _run_perturbation(self,airplane,perturbations,tag=None):
        """Perturbs the specified aircraft according to the perturbations dict."""
        perturbed = copy.deepcopy(self.trimmed_dict)
        if tag is None:
            tag = str(random.randint(0,9999))

        for key in perturbations: # The key specifies the variable to perturb

            if key in "dxdydz": # Position
                perturbed = self._perturb_position(perturbed,key,perturbations[key],airplane)

            elif key in "aileronelevatorrudder": # Control input
                perturbed = self._perturb_control(perturbed,key,perturbations[key],airplane)

            elif key == "dp": # Roll
                perturbed = self._perturb_roll(perturbed,perturbations[key],airplane)
        
            elif key == "dq": # Pitch
                perturbed = self._perturb_pitch(perturbed,perturbations[key],airplane)

            elif key == "dr": # Yaw
                perturbed = self._perturb_yaw(perturbed,perturbations[key],airplane)

        perturbed_file = tag+"perturbed.json"
        with open(perturbed_file, 'w', newline='\r\n') as dump_file:
            json.dump(perturbed,dump_file,indent=4)

        os.system("./Machup.out "+perturbed_file)

        return perturbed_file

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

    def _get_perturbed_forces_and_moments(self,args):
        """Perturbs the specified aircraft from trim and extracts the forces and moments."""
        airplane,perturbations = args

        # Perturb and extract forces and moments
        filename = self._run_perturbation(airplane,perturbations)
        lead_FM,tail_FM = self.get_forces_and_moments(filename)

        # Clean up Machup files
        os.system("rm "+filename)
        os.system("rm "+filename.replace(".json","_forces.json"))

        if airplane == "lead":
            return lead_FM
        else:
            return tail_FM

    def calc_cent_diff(self,args):
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

        derivs = (fwd_FM-bwd_FM)/(h)

        return derivs

    def run_derivs(self):
        """Find the matrix of derivatives of aerodynamic forces and moments with respect to position, orientation, and control input."""

        # Distribute calculations among processes
        deriv_list = ["dx","dy","dz","dp","dq","dr","aileron","elevator","rudder"]
        arg_list = [(deriv,"tail") for deriv in deriv_list]
        with mp.Pool(mp.cpu_count()) as pool:
            F_pbar_u = pool.map(self.calc_cent_diff,arg_list)

        # Format derivative matrices
        for i in range(len(F_pbar_u)):
            F_pbar_u[i] = F_pbar_u[i].flatten()

        self.F_pbar = np.asarray(F_pbar_u[:6])
        self.F_u = np.asarray(F_pbar_u[6:])

    def plot_grid_convergence(self,grids,separation_vec,trim_iterations=1):
        state_derivs = []
        control_derivs = []

        # Run derivatives for each grid size
        for grid in grids:
            self.grid = grid
            self._combine()
            situ.apply_separation_and_trim(separation_vec,iterations=trim_iterations)
            situ.run_derivs()
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
                plt.semilogx(grids,state_derivs[:,i,j],'x-')
                plt.xlabel("Grid Points")
                deriv_name = "d"+self.FM_names[j]+"/d"+self.state_names[i]
                plt.ylabel(deriv_name)
                plt.savefig(plot_dir+deriv_name.replace("/","_"))
                
        # Control derivatives
        shape = control_derivs[0].shape
        for i in range(shape[0]):
            for j in range(shape[1]):
                plt.figure()
                plt.semilogx(grids,control_derivs[:,i,j],'x-')
                plt.xlabel("Grid Points")
                deriv_name = "d"+self.FM_names[j]+"/dd"+self.control_names[i]
                plt.ylabel(deriv_name)
                plt.savefig(plot_dir+deriv_name.replace("/","_"))

    def _get_normal_perturbations(self,dispersions):
        """Generates a dict of normally distributed perturbations in the variables specified by dispersions."""
        perturbations = {}

        for key in dispersions:
            perturbations[key] = np.random.normal(0.0,dispersions[key])

        return perturbations

    def run_monte_carlo(self,separation_vec,airplane,N_samples,dispersions,grid=40):
        """Runs a Monte Carlo simulation to determine the dispersion of forces and moments."""
        self.grid = grid
        self._combine()
        self.apply_separation_and_trim(separation_vec,iterations=2)

        # Apply to multiprocessing
        start_time = time.time()
        arg_list = [(airplane,self._get_normal_perturbations(dispersions)) for i in range(N_samples)]
        with mp.Pool(mp.cpu_count()) as pool:
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
        var_of_var = np.zeros(shape)
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
                fit = stats.norm.pdf(x_space,x_mean,x_std)
                plt.plot(x_space,fit,"r-",label="Normal PDF")

                # Plot confidence interval of standard deviation
                x_std_upper = x_std_info.minmax[1]
                x_std_lower = x_std_info.minmax[0]
                upper_fit = stats.norm.pdf(x_space,x_mean,x_std_upper)
                lower_fit = stats.norm.pdf(x_space,x_mean,x_std_lower)
                plt.plot(x_space,upper_fit,"r--")
                plt.plot(x_space,lower_fit,"r--")

                # Format and save
                plt.xlabel(name)
                plt.legend()
                plt.savefig(plot_dir+"/"+name)

        return end_time-start_time

    def run_lin_cov(self,separation_vec,airplane,dispersions,grid=40):
        """Runs linear covariance analysis on the specified airplane using the given dispersions."""
        self.grid = grid
        self._combine()
        self.apply_separation_and_trim(separation_vec,iterations=2)

        # Calculate derivatives
        self.run_derivs()

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
    situ = TailingScenario(lead,tail,grid=40)
    #situ.apply_separation_and_trim(r_CG,iterations=2)
    #situ.run_derivs()
    #print("\nFp:\n{0}".format(situ.F_pbar))
    #print("\nFu:\n{0}".format(situ.F_u))

    # Check grid convergence of derivatives
    grids = [10,20,35,60,110,200]
    #situ.plot_grid_convergence(grids,r_CG)

    # Run Monte Carlo simulation
    dispersions = {
        "dx": 1.0,
        "dy": 1.0,
        "dz": 1.0,
        "dp": 0.05,
        "dq": 0.05,
        "aileron": 0.01,
        "elevator": 0.01,
        "rudder": 0.01
    }
    MC_exec_time = situ.run_monte_carlo(r_CG,"tail",1000,dispersions)
    print("Monte Carlo took {0} s to run.".format(MC_exec_time))
    print(situ.FM_dispersions)

    # Run LinCov