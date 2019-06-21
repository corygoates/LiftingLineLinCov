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
    perturbations in yaw. All perturbations are with respect to the wind frame.
    
    Params:
    
    lead_filename (str):
        A .json file which is the MachUp input for the leading aircraft.
        
    tail_filename (str):
        A .json file which is the MachUp input for the tailing aircraft.
        
    dx (float, optional):
        Increment to be used in calculating derivatives with respect to position. Defaults to 1.

    dtheta (float, optional):
        Increment to be used in calculating derivatives with respect to orientation or control deflection.
        Defaults to 1.

    default_grid (int, optional):
        Number of control points to be used on each lifting surface in numerical lifting line. Defaults to 100.
        Can be specified for each function called, but this default will be used if no specification is made.

    verbose (bool, optional):
        Flag to specify whether to display the output of MachUp. Defaults to False.
    """

    def __init__(self,lead_filename,tail_filename,**kwargs):
        self._lead_filename = lead_filename
        self._tail_filename = tail_filename
        self._dx = kwargs.get("dx",1.0)
        self._dtheta = kwargs.get("dtheta",1.0)
        self._default_grid = kwargs.get("default_grid",100)
        self._grid = copy.deepcopy(self._default_grid)
        self._verbose = kwargs.get("verbose",False)

        self._FM_names = ["Fx","Fy","Fz","Mx","My","Mz"]
        # These specify the order in which pertrubations are applied. The current formulation, to ensure grid resolution, applies pitch then roll.
        self._state_names = ["x","y","z","theta","phi"]#,"psi"]
        self._control_names = ["aileron","elevator"]#,"rudder"]

        self._DOF_names = ["d"+name for name in self._state_names]
        for name in self._control_names:
            self._DOF_names.append(name)

        self._combine()

        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman"]+plt.rcParams["font.serif"]

    def _combine(self):
        """Combines two airplane .json files."""

        # Load initial files
        with open(self._lead_filename,'r') as lead_file_handle:
            self._lead_dict = json.load(lead_file_handle)

        with open(self._tail_filename,'r') as tail_file_handle:
            self._tail_dict = json.load(tail_file_handle)

        # Create new copy
        self._combined_dict = copy.deepcopy(self._lead_dict)

        # Alter some key values
        del self._combined_dict["solver"]["convergence"] # Use default convergence to avoid FORTRAN not being able to read Python-exported floating points
        self._combined_dict["solver"]["type"] = "nonlinear"
        self._combined_dict["controls"] = {}
        self._combined_dict["wings"] = {}
        self._combined_dict["airfoil_DB"] = "./AirfoilDatabase"
        self._combined_dict["run"] = {"forces":""}
        self._combined_dict["plane"]["name"] = "lead_tail_combo"
        self._combined_dict["condition"]["alpha"] = 0.0
        self._combined_dict["condition"]["beta"] = 0.0

        # Copy weights
        del self._combined_dict["condition"]["W"]
        self._combined_dict["condition"]["W_lead"] = copy.copy(self._lead_dict["condition"]["W"])
        self._combined_dict["condition"]["W_tail"] = copy.copy(self._tail_dict["condition"]["W"])

        # Store angles of attack as individual elevation angles
        self._combined_dict["thetas"] = {}
        self._combined_dict["thetas"]["lead"] = copy.copy(self._lead_dict["condition"]["alpha"])
        self._combined_dict["thetas"]["tail"] = copy.copy(self._tail_dict["condition"]["alpha"])

        # Store bank angles
        self._combined_dict["phis"] = {}
        self._combined_dict["phis"]["lead"] = 0.0
        self._combined_dict["phis"]["tail"] = 0.0

        # Copy lead controls
        for key in self._lead_dict["controls"]:
            new_key = "lead_"+key
            self._combined_dict["controls"][new_key] = copy.copy(self._lead_dict["controls"][key])

        # Copy lead surfaces
        for key in self._lead_dict["wings"]:
            #if "v" in key: # Skip vertical stabilizers for now to avoid grid convergence issues
            #    continue
            new_key = "lead_"+key
            self._combined_dict["wings"][new_key] = copy.copy(self._lead_dict["wings"][key])
            self._combined_dict["wings"][new_key]["grid"] = self._grid
            for control_key in self._combined_dict["wings"][new_key]["control"]["mix"]:
                self._combined_dict["wings"][new_key]["control"]["mix"] = {"lead_"+control_key:1.0}

        # Copy tail controls
        for key in self._tail_dict["controls"]:
            new_key = "tail_"+key
            self._combined_dict["controls"][new_key] = copy.copy(self._tail_dict["controls"][key])

        # For tailing lifting surfaces, split double surfaces into two surfaces
        for key in self._tail_dict["wings"]:
            if "v" in key: # Skip vertical stabilizers for now to avoid grid convergence issues
                continue
            new_key = "tail_"+key

            if self._tail_dict["wings"][key]["side"] == "both": # Double sided surface
                # Left side
                self._combined_dict["wings"][new_key+"_left"] = copy.deepcopy(self._tail_dict["wings"][key])
                self._combined_dict["wings"][new_key+"_left"]["side"] = "left"
                for control_key in self._combined_dict["wings"][new_key+"_left"]["control"]["mix"]:
                    self._combined_dict["wings"][new_key+"_left"]["control"]["mix"] = {"tail_"+control_key:1.0}
                    
                # Right Side
                self._combined_dict["wings"][new_key+"_right"] = copy.deepcopy(self._tail_dict["wings"][key])
                self._combined_dict["wings"][new_key+"_right"]["side"] = "right"
                for control_key in self._combined_dict["wings"][new_key+"_right"]["control"]["mix"]:
                    self._combined_dict["wings"][new_key+"_right"]["control"]["mix"] = {"tail_"+control_key:1.0}

            else: # Single sided surface
                self._combined_dict["wings"][new_key] = self._tail_dict["wings"][key]
                for control_key in self._combined_dict["wings"][new_key]["control"]["mix"]:
                    self._combined_dict["wings"][new_key]["control"]["mix"] = {"tail_"+control_key:1.0}

        # Apply grid resolution to all lifting surfaces
        for wing in self._combined_dict["wings"]:
            self._combined_dict["wings"][wing]["grid"] = self._grid

    def _apply_new_grid(self,grid=None):
        """Sets the number of grid points for each lifting surface. Will not alter the original configuration."""

        # Assign default grid if needs be
        if grid is None:
            grid = self._default_grid
        self._grid = grid

        if hasattr(self,"trimmed_dict"):
            # Apply to trimmed dict
            for wing in self._trimmed_dict["wings"]:
                self._trimmed_dict["wings"][wing]["grid"] = self._grid
        
    def _apply_theta_de(self,theta,de,airplane):
        """Applies an elevation angle and elevator deflection to the specified airplane."""

        # Apply elevator deflection
        self._trimmed_dict["controls"][airplane+"_elevator"]["deflection"] = de

        # Apply theta
        for wing in self._trimmed_dict["wings"]:
            if airplane in wing:
                x0 = copy.copy(self._combined_dict["wings"][wing]["connect"]["dx"])
                z0 = copy.copy(self._combined_dict["wings"][wing]["connect"]["dz"])
                C = np.cos(np.radians(theta))
                S = np.sin(np.radians(theta))
                x1 = x0*C+z0*S
                z1 = z0*C-x0*S
                if airplane == "tail":
                    x1 += self._r[0]
                    z1 += self._r[2]
                self._trimmed_dict["wings"][wing]["connect"]["dx"] = copy.copy(x1)
                self._trimmed_dict["wings"][wing]["connect"]["dz"] = copy.copy(z1)

                if "v" in wing: # Vertical stabilizer
                    self._trimmed_dict["wings"][wing]["sweep"] = self._combined_dict["wings"][wing]["sweep"]+theta
                else: # Horizontal surface
                    self._trimmed_dict["wings"][wing]["mounting_angle"] = self._combined_dict["wings"][wing]["mounting_angle"]+theta

        # Store theta
        self._trimmed_dict["thetas"][airplane] = theta
        
        # Save output
        with open(self.trimmed_filename,'w') as dump_file_handle:
            json.dump(self._trimmed_dict,dump_file_handle,indent=4)

    def _run_machup(self,input_filename):
        """Interfaces with the Machup executable to run the specified config file."""
        try:
            # Run executable
            print("MachUp is executing {0}...".format(input_filename))
            if self._verbose:
                completed = sp.run(["./Machup.out",input_filename])
            else:
                with open(os.devnull,'w') as FNULL:
                    completed = sp.run(["./Machup.out",input_filename],stdout=FNULL,stderr=sp.STDOUT)

            # Unexpected completion
            if completed.returncode < 0:
                print("MachUp terminated with code {0} after trying to execute {1}.".format(-completed.returncode,input_filename))

            # Successful completion
            else:
                print("MachUp successfully completed executing {0}.".format(input_filename))

        # Catch any errors
        except OSError as e:
            raise RuntimeError("MachUp execution failed while executing {0} due to {1}.".format(input_filename,e))

    def _get_trim_residuals(self,theta,de,airplane):
        """Returns L-W and m for the given aircraft in the given state."""

        # Apply elevation angle and elevator deflection and run
        self._apply_theta_de(theta,de,airplane)
        self._run_machup(self.trimmed_filename)
        lead_FM,tail_FM = self._get_forces_and_moments(self.trimmed_filename,body_ref=False)

        # Calculate residuals
        # We do this in the wind frame to simplify calculations (W remains aligned with wind z-axis)
        if airplane == "tail":
            R_F_z = np.asscalar(tail_FM[0,2])+self._combined_dict["condition"]["W_"+airplane]
            R_m_y = np.asscalar(tail_FM[1,1])
        else:
            R_F_z = np.asscalar(lead_FM[0,2])+self._combined_dict["condition"]["W_"+airplane]
            R_m_y = np.asscalar(lead_FM[1,1])

        return np.asarray([R_F_z,R_m_y])

    def _get_residuals_and_derivs(self,theta,de,airplane):
        """Returns the residuals and derivatives for the given trim state."""
        # This doesn't use self._calc_central_diff() in order to take full advantage of multiprocessing (5 at a time instead of 3 and then 2)
        # Apply to multiprocessing
        perturb_list = [{"dtheta":self._dtheta},{"dtheta":-self._dtheta},{"elevator":self._dtheta},{"elevator":-self._dtheta}]

        arg_list = [(airplane,perturbation,-1,False) for perturbation in perturb_list]
        with mp.Pool() as pool:
            R_get = pool.apply_async(self._get_trim_residuals,(theta,de,airplane))
            FM_get = pool.map_async(self._get_perturbed_forces_and_moments,arg_list)
            R = R_get.get()
            FM = FM_get.get()

        # Calculate central difference
        h = 2*self._dtheta
        derivs = []
        derivs.append((FM[0]-FM[1])/h)
        derivs.append((FM[2]-FM[3])/h)

        return R, derivs

    def _trim(self,theta_init,de_init,airplane,convergence):
        """Uses Newton's method to solve for trim."""
        theta0 = copy.deepcopy(theta_init)
        de0 = copy.deepcopy(de_init)
        print("\nTrimming "+airplane+" aircraft.")

        # Calculate derivatives and residuals
        R, derivs = self._get_residuals_and_derivs(theta0,de0,airplane)

        # Output progress
        print("Trim residuals:\n    Fz: {0}\n    My: {1}".format(R[0],R[1]))
    
        while (abs(R)>convergence).any(): # If the trim conditions have not yet converged

            # Arrange Jacobian
            J = np.zeros((2,2))
            J[0,0] = derivs[0][0,2] # dFz/da
            J[0,1] = derivs[1][0,2] # dFz/dde
            J[1,0] = derivs[0][1,1] # dmy/da
            J[1,1] = derivs[1][1,1] # dmy/dde

            # Correct pitch and elevator deflection
            corrector = np.asarray(np.linalg.inv(J)*np.asmatrix(R).T)
            theta1 = np.asscalar(theta0-corrector[0])
            de1 = np.asscalar(de0-corrector[1])

            # Calculate derivatives and residuals
            R, derivs = self._get_residuals_and_derivs(theta1,de1,airplane)

            # Output progress
            print("Trim residuals:\n    Fz: {0}\n    My: {1}".format(R[0],R[1]))

            # Update for next iteration
            theta0 = theta1
            de0 = de1

        self._apply_theta_de(theta0,de0,airplane) # Because this was done in another process

    def _apply_separation_and_trim(self,separation_vec,iterations=1,grid=None,convergence=1e-8,export_stl=False):
        """Separates the two aircraft according to separation_vec and trims both.
        This will leave alpha and beta as 0.0, instead trimming using mounting
        angles. This will first trim the leading aircraft, then the tailing aircraft.
        This process can be repeated for finer trim results using the iterations
        parameter."""

        # Assign default grid if needs be
        if grid is None:
            grid = self._default_grid

        self._apply_new_grid(grid)

        # If the trimmed dict does not already exist, create it
        # This is done so that subsequent trimming operations have a better starting point.
        if not hasattr(self,"trimmed_dict"):
            self._trimmed_dict = copy.deepcopy(self._combined_dict)

        # Apply separation
        self._r = separation_vec
        for wing in self._trimmed_dict["wings"]:
            if "tail" in wing:
                self._trimmed_dict["wings"][wing]["connect"]["dx"] = self._combined_dict["wings"][wing]["connect"]["dx"]+self._r[0]
                self._trimmed_dict["wings"][wing]["connect"]["dy"] = self._combined_dict["wings"][wing]["connect"]["dy"]+self._r[1]
                self._trimmed_dict["wings"][wing]["connect"]["dz"] = self._combined_dict["wings"][wing]["connect"]["dz"]+self._r[2]
        
        # Dump info
        self.trimmed_filename = "trimmed.json"
        with open(self.trimmed_filename,'w') as dump_file_handle:
            json.dump(self._trimmed_dict,dump_file_handle,indent=4)

        # Trim each plane iteratively
        for _ in range(iterations):
            airplanes = ["lead","tail"]
            for airplane in airplanes:

                # Get an initial guess for the elevation angle and elevator deflection
                try:
                    theta_init = self._trimmed_dict["thetas"][airplane]
                except KeyError:
                    if airplane == "lead":
                        theta_init = self._lead_dict["condition"]["alpha"]
                    else:
                        theta_init = self._tail_dict["condition"]["alpha"]

                de_init = self._trimmed_dict["controls"][airplane+"_elevator"]["deflection"]

                # Trim
                self._trim(theta_init,de_init,airplane,convergence)

        # Export .stl file of the trimmed situation if specified
        if export_stl:
            stl_dict = copy.deepcopy(self._trimmed_dict)
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
                    x0 -= self._r[0]
                    z0 -= self._r[2]
                C = np.cos(np.radians(perturbation))
                S = np.sin(np.radians(perturbation))
                x1 = x0*C+z0*S
                z1 = -x0*S+z0*C
                if airplane == "tail":
                    x1 += self._r[0]
                    z1 += self._r[2]
                airplane_dict["wings"][wing]["connect"]["dx"] = copy.copy(x1)
                airplane_dict["wings"][wing]["connect"]["dz"] = copy.copy(z1)

                # Apply to angle
                if "v" in wing: # Vertical stabilizer
                    airplane_dict["wings"][wing]["sweep"] += perturbation
                else: # Horizontal surface
                    airplane_dict["wings"][wing]["mounting_angle"] += perturbation

        airplane_dict["thetas"][airplane] += perturbation

        return airplane_dict

    def _perturb_roll(self,airplane_dict,perturbation,airplane):
        """Perturbs the specified aircraft in roll."""
        for wing in airplane_dict["wings"]:
            if airplane in wing:

                # Apply to position
                y0 = copy.deepcopy(airplane_dict["wings"][wing]["connect"]["dy"])
                z0 = copy.deepcopy(airplane_dict["wings"][wing]["connect"]["dz"])
                if airplane == "tail":
                    y0 -= self._r[1]
                    z0 -= self._r[2]
                C = np.cos(np.radians(perturbation))
                S = np.sin(np.radians(perturbation))
                y1 = y0*C-z0*S
                z1 = y0*S+z0*C
                if airplane == "tail":
                    y1 += self._r[1]
                    z1 += self._r[2]
                airplane_dict["wings"][wing]["connect"]["dy"] = copy.copy(y1)
                airplane_dict["wings"][wing]["connect"]["dz"] = copy.copy(z1)

                # Apply to angle
                if airplane_dict["wings"][wing]["side"] == "right":
                    airplane_dict["wings"][wing]["dihedral"] -= perturbation
                else:
                    airplane_dict["wings"][wing]["dihedral"] += perturbation

        airplane_dict["phis"][airplane] += perturbation

        return airplane_dict

    def _run_perturbation(self,airplane,perturbations):
        """Perturbs the specified aircraft according to the perturbations dict."""
        perturbed = copy.deepcopy(self._trimmed_dict)

        # Create descriptive file tag
        tag_list = []
        for key in perturbations:
            tag_list.append(key+"{:.2f}".format(perturbations[key]))
        tag = "".join(tag_list)

        # Apply specified perturbations
        # Ensure position perturbations are applied first
        for key in perturbations: # The key specifies the variable to perturb
            if key in "dxdydz": # Position
                perturbed = self._perturb_position(perturbed,key,perturbations[key],airplane)

        # Once position perturbations are applied, apply all others
        for key in perturbations: # The key specifies the variable to perturb
            if key in "aileronelevatorrudder": # Control input
                perturbed = self._perturb_control(perturbed,key,perturbations[key],airplane)

            elif key == "dphi": # Roll
                perturbed = self._perturb_roll(perturbed,perturbations[key],airplane)
        
            elif key == "dtheta": # Pitch
                perturbed = self._perturb_pitch(perturbed,perturbations[key],airplane)

            elif key == "dpsi": # Yaw
                raise ValueError("Perturbations in yaw are not allowed.")

        # Store and run perturbed dict
        perturbed_file = tag+"perturbed.json"
        with open(perturbed_file, 'w') as dump_file:
            json.dump(perturbed,dump_file,indent=4)

        self._run_machup(perturbed_file)

        return perturbed_file

    def _transform_to_body_frame(self,psi,theta,phi,vec):
        """Transforms the given vector from wind frame to body frame."""
        T = np.zeros((3,3))

        # Find trigonometric relations
        C_phi = np.cos(np.radians(phi))
        S_phi = np.sin(np.radians(phi))
        C_theta = np.cos(np.radians(theta))
        S_theta = np.sin(np.radians(theta))
        C_psi = np.cos(np.radians(psi))
        S_psi = np.sin(np.radians(psi))

        # This is the transformation of a vector due to a pitch about the wind y-axis and a roll about the wind x-axis.
        # The commented-out calculations are the Euler angle transformation. These are not strictly Euler angles, so we use a different transformation.
        T[0,0] = C_theta#C_theta*C_psi
        T[0,1] = S_theta*S_phi#C_theta*S_psi
        T[0,2] = -S_theta*C_phi#-S_theta
        T[1,0] = 0#S_phi*S_theta*C_psi-C_phi*S_psi
        T[1,1] = C_phi#S_phi*S_theta*S_psi+C_phi*C_psi
        T[1,2] = S_phi#S_phi*C_theta
        T[2,0] = S_theta#C_phi*S_theta*C_psi+S_phi*S_psi
        T[2,1] = -C_theta*S_phi#C_phi*S_theta*S_psi-S_phi*C_psi
        T[2,2] = C_theta*C_phi#C_phi*C_theta

        return np.matmul(T,vec)

    def _get_forces_and_moments(self,filename,body_ref=True):
        """Extracts aerodynamic forces and moments from the specified MachUp output file."""

        # Define reference constants
        q_inf = 0.5*self._combined_dict["condition"]["density"]*self._combined_dict["condition"]["velocity"]**2
        S_w = self._combined_dict["reference"]["area"]
        l_ref_lon = self._combined_dict["reference"]["longitudinal_length"]
        l_ref_lat = self._combined_dict["reference"]["lateral_length"]

        # Load input file to determine orientation if the forces and moments need to be output in the body frame
        if body_ref:
            with open(filename,'r') as input_file:
                input_dict = json.load(input_file)
            
            lead_phi = input_dict["phis"]["lead"]
            lead_theta = input_dict["thetas"]["lead"]
            tail_phi = input_dict["phis"]["tail"]
            tail_theta = input_dict["thetas"]["tail"]

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
        mom_trans = np.cross(self._r,tail_FM[0])
        tail_FM[1] -= mom_trans

        # Transform to body frame
        if body_ref:
            lead_FM = self._transform_to_body_frame(0.0,lead_theta,lead_phi,tail_FM.T).T
            tail_FM = self._transform_to_body_frame(0.0,tail_theta,tail_phi,tail_FM.T).T

        return lead_FM,tail_FM

    def _get_perturbed_forces_and_moments(self,args):
        """Perturbs the specified aircraft from trim and extracts the forces and moments."""

        # Unpack arguments
        if len(args) is 3:
            airplane,perturbations,case_no = args
            body_ref = True
        elif len(args) is 4:
            airplane,perturbations,case_no,body_ref = args
        else:
            raise ValueError("Improper number of args passed to _get_perturbed_forces_and_moments.")

        # Print case number
        if case_no is not -1:
            print("Case No. {0}".format(case_no))

        # Update separation vector according to the perturbations
        for key in perturbations:
            if key == "dx":
                self._r[0] += perturbations[key]
            elif key == "dy":
                self._r[1] += perturbations[key]
            elif key == "dz":
                self._r[2] += perturbations[key]

        # Perturb
        filename = self._run_perturbation(airplane,perturbations)

        # Extract forces and moments
        lead_FM,tail_FM = self._get_forces_and_moments(filename,body_ref=body_ref)

        # Reset separation vector
        for key in perturbations:
            if key == "dx":
                self._r[0] -= perturbations[key]
            elif key == "dy":
                self._r[1] -= perturbations[key]
            elif key == "dz":
                self._r[2] -= perturbations[key]

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
            fwd_FM = self._get_perturbed_forces_and_moments((airplane,{wrt:self._dx},-1))
            bwd_FM = self._get_perturbed_forces_and_moments((airplane,{wrt:-self._dx},-1))
            h = 2*self._dx

        else: # Angular
            fwd_FM = self._get_perturbed_forces_and_moments((airplane,{wrt:self._dtheta},-1))
            bwd_FM = self._get_perturbed_forces_and_moments((airplane,{wrt:-self._dtheta},-1))
            h = 2*self._dtheta

        derivs = (fwd_FM-bwd_FM)/h

        return derivs

    def run_derivs(self,airplane,separation_vec,trim_iterations=2,grid=None,export_stl=False,trim=True):
        """Find the matrix of derivatives of aerodynamic forces and moments with respect to position, orientation, and control input."""

        # Trim if asked
        if trim:
            self._apply_separation_and_trim(separation_vec,iterations=trim_iterations,grid=grid,export_stl=export_stl)
        else:
            self._apply_new_grid(grid) # Just to make sure

        print("\nCalculating derivatives.")

        # Distribute calculations among processes
        deriv_list = ["d"+state_var for state_var in self._state_names]
        for control in self._control_names:
            deriv_list.append(control)

        arg_list = [(deriv,airplane) for deriv in deriv_list]
        with mp.Pool() as pool:
            F_xc = pool.map(self._calc_cent_diff,arg_list)

        # Format derivative matrices
        F_xc = np.asarray(F_xc)
        num_states = len(self._state_names)

        self.Q_x = np.asarray(F_xc[:num_states,0]) # df/dx
        self.Q_u = np.asarray(F_xc[num_states:,0]) # df/du

        self.R_x = np.asarray(F_xc[:num_states,1]) # dm/dx
        self.R_u = np.asarray(F_xc[num_states:,1]) # dm/du

    def plot_derivative_convergence(self,grids,separation_vec,trim_iterations=2,trim_once=True,trim_grid=None):
        """Plots each derivative as a function of grid points used to show convergence."""

        # Assign default grid if needs be
        if trim_grid is None:
            trim_grid = self._default_grid

        # Initialize storage lists
        Q_x_list = []
        Q_u_list = []
        R_x_list = []
        R_u_list = []

        # Run derivatives for each grid size
        if trim_once:
            self._apply_separation_and_trim(separation_vec,iterations=trim_iterations,grid=trim_grid)
        for grid in grids:
            print("\nCalculating derivatives at a grid size of {0}".format(grid))
            if trim_once:
                situ.run_derivs("tail",separation_vec,grid=grid,trim=False)
            else:
                situ.run_derivs("tail",separation_vec,trim_iterations=trim_iterations,grid=grid)

            Q_x_list.append(self.Q_x)
            Q_u_list.append(self.Q_u)
            R_x_list.append(self.R_x)
            R_u_list.append(self.R_u)

        Q_x = np.asarray(Q_x_list)
        Q_u = np.asarray(Q_u_list)
        R_x = np.asarray(R_x_list)
        R_u = np.asarray(R_u_list)

        # Create plot folder
        plot_dir = "./DerivativeConvergencePlots/"
        if not os.path.exists(plot_dir):
            os.mkdir(plot_dir)

        # Plot convergence results
        # Force derivatives with respect to state
        shape = Q_x[0].shape
        for i in range(shape[0]):
            for j in range(shape[1]):
                plt.figure()
                plt.semilogx(grids,Q_x[:,i,j],'kx--')
                plt.xlabel("Grid Points")
                deriv_name = "d"+self._FM_names[j]+"/d"+self._state_names[i]
                plt.ylabel(deriv_name)
                plt.savefig(plot_dir+deriv_name.replace("/","_"),bbox_inches='tight',format='svg')
                plt.close()

        # Moment derivatives with respect to state
        shape = R_x[0].shape
        for i in range(shape[0]):
            for j in range(shape[1]):
                plt.figure()
                plt.semilogx(grids,R_x[:,i,j],'kx--')
                plt.xlabel("Grid Points")
                deriv_name = "d"+self._FM_names[j+3]+"/d"+self._state_names[i]
                plt.ylabel(deriv_name)
                plt.savefig(plot_dir+deriv_name.replace("/","_"),bbox_inches='tight',format='svg')
                plt.close()
                
        # Force derivatives with respect to controls
        shape = Q_u[0].shape
        for i in range(shape[0]):
            for j in range(shape[1]):
                plt.figure()
                plt.semilogx(grids,Q_u[:,i,j],'kx--')
                plt.xlabel("Grid Points")
                deriv_name = "d"+self._FM_names[j]+"/dd"+self._control_names[i][0]
                plt.ylabel(deriv_name)
                plt.savefig(plot_dir+deriv_name.replace("/","_"),bbox_inches='tight',format='svg')
                plt.close()
                
        # Force derivatives with respect to controls
        shape = R_u[0].shape
        for i in range(shape[0]):
            for j in range(shape[1]):
                plt.figure()
                plt.semilogx(grids,R_u[:,i,j],'kx--')
                plt.xlabel("Grid Points")
                deriv_name = "d"+self._FM_names[j+3]+"/dd"+self._control_names[i][0]
                plt.ylabel(deriv_name)
                plt.savefig(plot_dir+deriv_name.replace("/","_"),bbox_inches='tight',format='svg')
                plt.close()

    def plot_perturbation_convergence(self,separation_vec,airplane,perturbation,grids,trim_iterations=2,trim_once=True,trim_grid=None):
        """Plots the grid convergence of the forces and moments generated by a given perturbation."""

        # Assign default grid if needs be
        if trim_grid is None:
            trim_grid = self._default_grid

        # Run perturbation at each grid level
        FM_list = []
        theta_lead_list = []
        theta_tail_list = []

        # If only trimming once
        if trim_once:
            self._apply_separation_and_trim(separation_vec,iterations=trim_iterations,grid=trim_grid)
        for grid in grids:
            self._apply_new_grid(grid)

            # If trimming at each grid level
            if not trim_once:
                self._apply_separation_and_trim(separation_vec,iterations=trim_iterations,grid=grid)

            FM_list.append(self._get_perturbed_forces_and_moments((airplane,perturbation,-1)))
            theta_lead_list.append(self._trimmed_dict["thetas"]["lead"])
            theta_tail_list.append(self._trimmed_dict["thetas"]["tail"])

        # Create directories
        var = list(perturbation.keys())[0]
        parent_dir = "./FMConvergencePlots"
        if not os.path.exists(parent_dir):
            os.mkdir(parent_dir)
        if trim_once:
            plot_dir = parent_dir+"/TrimmedOnceWRT"+var
        else:
            plot_dir = parent_dir+"/TrimmedAtEachWRT"+var

        if not os.path.exists(plot_dir):
            os.mkdir(plot_dir)

        # Plot forces and moments as a function of grid resolution
        FM = np.asarray(FM_list)
        shape = FM[0].shape
        for i in range(shape[0]):
            for j in range(shape[1]):
                plt.figure()
                plt.semilogx(grids,FM[:,i,j],"kx--")
                name = self._FM_names[i*shape[1]+j]
                plt.xlabel("Grid Points")
                plt.ylabel(name)
                plt.savefig(plot_dir+"/"+name,bbox_inches='tight',format='svg')
                plt.close()

        # Angles of attack
        plt.figure()
        plt.semilogx(grids,theta_lead_list,"kx--")
        plt.xlabel("Grid Points")
        plt.ylabel("Lead elevation angle")
        plt.savefig(plot_dir+"/Leadtheta",bbox_inches='tight',format='svg')
        plt.close()

        plt.figure()
        plt.semilogx(grids,theta_tail_list,"kx--")
        plt.xlabel("Grid Points")
        plt.ylabel("Tail elevation angle")
        plt.savefig(plot_dir+"/Tailtheta",bbox_inches='tight',format='svg')
        plt.close()

    def _get_normal_perturbations(self,variances):
        """Generates a dict of normally distributed perturbations in the variables specified by variances."""
        perturbations = {}

        # Get normally distributed variable for each perturbation
        for key in variances:
            perturbations[key] = np.random.normal(0.0,np.sqrt(variances[key]))

        return perturbations

    def run_monte_carlo(self,separation_vec,airplane,N_samples,variances,grid=None,trim=True,trim_iterations=2):
        """Runs a Monte Carlo simulation to determine the covariance of forces and moments.
           Note: If a new separation_vec is specified, then trim must be set to true."""

        start_time = time.time()
        # Assign default grid if needs be
        if grid is None:
            grid = self._default_grid
        if trim:
            self._apply_separation_and_trim(separation_vec,iterations=trim_iterations,grid=grid)
        else:
            self._apply_new_grid(grid)

        # Apply to multiprocessing
        arg_list = [(airplane,self._get_normal_perturbations(variances),i) for i in range(N_samples)]
        with mp.Pool() as pool:
            FM_samples = pool.map(self._get_perturbed_forces_and_moments,arg_list)

        # Calculate input covariance
        self.P_xx_uu = np.zeros((len(variances.keys()),len(variances.keys())))

        for arg_set in arg_list:
            xu = np.zeros((len(variances.keys()),1))
            for i,key in enumerate(variances):
                xu[i,0] = arg_set[1][key]

            self.P_xx_uu += np.matmul(xu,xu.T)

        self.P_xx_uu = self.P_xx_uu/(N_samples-1)

        # Get nominal forces and moments
        #if trim: # If it's been trimmed at this grid level, just read in the file
        #    _,FM_nom = self._get_forces_and_moments(self.trimmed_filename)
        #else:
        FM_nom = self._get_perturbed_forces_and_moments(("tail",{},-1))
        F_nom = FM_nom[0].flatten()
        M_nom = FM_nom[1].flatten()

        # Determine force and moment covariances
        self.P_ff_MC = np.zeros((3,3))
        self.P_mm_MC = np.zeros((3,3))

        for FM_sample in FM_samples:
            self.P_ff_MC += np.matmul((FM_sample[0]-F_nom).reshape((3,1)),(FM_sample[0]-F_nom).reshape((1,3)))
            self.P_mm_MC += np.matmul((FM_sample[1]-M_nom).reshape((3,1)),(FM_sample[1]-M_nom).reshape((1,3)))

        self.P_ff_MC = self.P_ff_MC/(N_samples-1)
        self.P_mm_MC = self.P_mm_MC/(N_samples-1)

        end_time = time.time()

        # Create plot storage directory
        plot_dir = "./MCDispersions"
        if not os.path.exists(plot_dir):
            os.mkdir(plot_dir)
        
        # Plot covariances in each force and moment
        FM_samples = np.asarray(FM_samples)
        shape = FM_samples[0].shape
        self.MC_dispersions = np.zeros(shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                plt.figure()
                name = self._FM_names[i*shape[1]+j]
                
                # Plot histogram
                x = FM_samples[:,i,j].flatten()
                plt.hist(x,density=True,label=name,color="gray",lw=0)

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
                plt.savefig(plot_dir+"/"+name,bbox_inches='tight',format='svg')
                plt.close()

        return end_time-start_time

    def run_lin_cov(self,separation_vec,airplane,input_variance,trim_iterations=2,grid=None,trim=True):
        """Runs linear covariance analysis on the specified airplane using the given variances.
           Note: If a new separation_vec is specified, then trim must be set to true."""

        # Calculate derivatives
        start_time = time.time()
        self.run_derivs(airplane,separation_vec,trim_iterations=trim_iterations,grid=grid,trim=trim)

        # Create covariance matrices
        P_xx = np.eye(len(self._state_names))
        P_uu = np.eye(len(self._control_names))
        
        for i,name in enumerate(self._state_names):
            P_xx[i,i] *= input_variance["d"+name]

        for i,name in enumerate(self._control_names):
            P_uu[i,i] *= input_variance[name]

        # Propagate
        self.P_ff_LC = self.Q_x.T.dot(P_xx).dot(self.Q_x)+self.Q_u.T.dot(P_uu).dot(self.Q_u)
        self.P_mm_LC = self.R_x.T.dot(P_xx).dot(self.R_x)+self.R_u.T.dot(P_uu).dot(self.R_u)
        end_time = time.time()
        return end_time-start_time

    def calc_MC_LC_error(self):
        """Calculates the percent error between the covariances predicted by MC and LC."""
        self.P_ff_error = abs(self.P_ff_LC-self.P_ff_MC)/abs(self.P_ff_MC)
        self.P_mm_error = abs(self.P_mm_LC-self.P_mm_MC)/abs(self.P_mm_MC)
    
    def print_nonlinear_coupling(self,r_CG,trim_iterations=1):
        """Displays how each pair of DOFs might couple."""

        # Setup
        self._apply_separation_and_trim(r_CG,iterations=trim_iterations)
        trim_FM = self._get_perturbed_forces_and_moments(("tail",{},-1))

        # Analyze each pair of DOFs
        for DOF0 in self._DOF_names:
            for DOF1 in self._DOF_names:
                if DOF0 is DOF1:
                    continue

                print("\n-----{0} and {1}-----".format(DOF0,DOF1))
                DOF0_FM = self._get_perturbed_forces_and_moments(("tail",{DOF0:1.0},-1))
                DOF1_FM = self._get_perturbed_forces_and_moments(("tail",{DOF1:1.0},-1))
                comb_FM = self._get_perturbed_forces_and_moments(("tail",{DOF0:1.0,DOF1:1.0},-1))
                print("\nPerturbed {0}: \n{1}".format(DOF0,DOF0_FM))
                print("\nPerturbed {0}: \n{1}".format(DOF1,DOF1_FM))
                
                # If this is truly linear, the result of perturbing both should be the linear combination of the original results
                predicted_FM = DOF0_FM+DOF1_FM-trim_FM
                print("\nPredicted perturbed {0} and {1}: \n{2}".format(DOF0,DOF1,predicted_FM))
                print("\nActual perturbed {0} and {1}: \n{2}".format(DOF0,DOF1,comb_FM))
                error = abs(predicted_FM-comb_FM)/abs(comb_FM)
                print("\nPrediction error perturbed {0} and {1}: \n{2}".format(DOF0,DOF1,error))

    def plot_mapping_linearity(self,r_CG,variances,trim_iterations=1,num_points=20):
        """Plots each force and moment as a function of each perturbation. Will plot out to 3-sigma."""

        # Setup
        self._apply_separation_and_trim(r_CG,iterations=trim_iterations)
        plot_dir = "./LinearityPlots"
        if not os.path.exists(plot_dir):
            os.mkdir(plot_dir)

        # Plot each DOF perturbed from trim
        for DOF in self._DOF_names:
            x_space = np.linspace(-3*np.sqrt(variances[DOF]),3*np.sqrt(variances[DOF]),num_points)

            # Get forces and moments
            arg_list = [("tail",{DOF:x},-1) for x in x_space]
            with mp.Pool() as pool:
                FM = pool.map(self._get_perturbed_forces_and_moments,arg_list)
            FM = np.asarray(FM)

            # Generate and store plots
            shape = FM[0].shape
            for i in range(shape[0]):
                for j in range(shape[1]):
                    plt.figure()
                    plt.plot(x_space,FM[:,i,j],"kx--")
                    plt.xlabel(DOF)
                    FM_name = self._FM_names[i*3+j]
                    plt.ylabel(FM_name)
                    plt.savefig(plot_dir+"/"+FM_name+"_"+DOF,bbox="tight",format="svg")
                    plt.close()

    def check_linear_approximations(self,r_CG,variances,trim_iterations=1):
        """Randomly generates perturbations according to variances.
           Compares the forces and moments predicted by the linear
           derivatives with those calculated by NLL."""

        # Calculate the derivatives and trim state
        self.run_derivs("tail",r_CG,trim_iterations=trim_iterations)
        print("Qx:\n{0}".format(self.Q_x))
        print("Qu:\n{0}".format(self.Q_u))
        print("Rx:\n{0}".format(self.R_x))
        print("Ru:\n{0}".format(self.R_u))
        trim_FM = self._get_perturbed_forces_and_moments(("tail",{},-1))
        
        # While the user still wants to
        while True:

            # Get random perturbations
            perturbation = self._get_normal_perturbations(variances)

            # Arrange in vectors
            p_x = np.zeros((len(self._state_names),1))
            p_u = np.zeros((len(self._control_names),1))
            for i,name in enumerate(self._state_names):
                p_x[i] = perturbation.get("d"+name,0.0)
            for i,name in enumerate(self._control_names):
                p_u[i] = perturbation.get(name,0.0)

            # Predict forces and moments
            F = np.matmul(self.Q_x.T,p_x)+np.matmul(self.Q_u.T,p_u)
            M = np.matmul(self.R_x.T,p_x)+np.matmul(self.R_u.T,p_u)
            FM_pred = np.concatenate((F.T,M.T),axis=0)+trim_FM
            
            # Calculate actuals
            FM_actual = self._get_perturbed_forces_and_moments(("tail",perturbation,-1))

            # Output
            print("\nPerturbations:\n{0}".format(perturbation))
            print("Predicted:\n{0}".format(FM_pred))
            print("Actual:\n{0}".format(FM_actual))
            print("Error:\n{0}".format(abs(FM_pred-FM_actual)/abs(FM_actual)))

            user_input = input("Type [q] to quit, [Enter] to continue:")
            if user_input is "q":
                break

    def check_roll(self):
        """Cool stuff."""
        # Make sure perturbations are doing what they should
        files = []
        for roll_angle in np.linspace(0,1800,10):
            files.append(situ._run_perturbation("tail",{"dtheta":30,"dphi":roll_angle}))

        with open(files[0],'r') as base_file:
            stacked_dict = json.load(base_file)

        for i in range(1,len(files)):
            with open(files[i].replace("_forces.json",".json")) as next_file:
                next_dict = json.load(next_file)

            for wing in next_dict["wings"]:
                if "tail" in wing:
                    stacked_dict["wings"][wing+str(i)] = copy.deepcopy(next_dict["wings"][wing])

        stacked_dict["run"] = {"stl":""}
        stacked_filename = "stacked.json"
        with open(stacked_filename,'w') as stacked_file:
            json.dump(stacked_dict,stacked_file)

        situ._run_machup(stacked_filename)

if __name__=="__main__":

    # Clean up old files
    sp.run(["rm","*perturbed*"])
    sp.run(["rm","*trim*"])

    # Define displacement of trailing aircraft
    r_CG = [-200,0,50]

    # Initialize scenario
    lead = "./IndividualModels/C130.json"
    tail = "./IndividualModels/ALTIUSjr.json"
    situ = TailingScenario(lead,tail,default_grid=40)

#    # Run sensitivities at the trim state
#    situ.run_derivs("tail",r_CG,export_stl=True,trim_iterations=1)
#    print("Qx:\n{0}".format(situ.Q_x))
#    print("Qu:\n{0}".format(situ.Q_u))
#    print("Rx:\n{0}".format(situ.R_x))
#    print("Ru:\n{0}".format(situ.R_u))
#
#    # Check force and moment grid convergence
#    grid_floats = np.logspace(1,2.5,10)
#    grids = list(map(int,grid_floats))
#    perturbings = [{"dtheta": 0.1},{"dphi": 0.1},{"dx": 0.1},{"dy": 0.1},{"dz": 0.1}]
#    for perturb in perturbings:
#        situ.plot_perturbation_convergence(r_CG,"tail",perturb,grids)
#
#    # Check grid convergence of derivatives
#    situ.plot_derivative_convergence(grids,r_CG)
#
#    # Observe nonlinear coupling
#    situ.print_nonlinear_coupling(r_CG,trim_iterations=0)
#
    # Run Monte Carlo simulation
    input_variance = {
        "dx": 9.0,
        "dy": 9.0,
        "dz": 36.0,
        "dtheta": 1.0,#25.0,
        "dphi": 25.0,
        "aileron": 16.0,
        "elevator": 16.0
    }
    N_MC_samples = 1000
    MC_exec_time = situ.run_monte_carlo(r_CG,"tail",N_MC_samples,input_variance,trim_iterations=1)

    # Run LinCov
    LC_exec_time = situ.run_lin_cov(r_CG,"tail",input_variance,trim=False)
#
#    # Check linear predictions
#    situ.check_linear_approximations(r_CG,input_variance)
#
#    # Check linearity
#    situ.plot_mapping_linearity(r_CG,input_variance,num_points=20)

    # Compare MC and LC results
    situ.calc_MC_LC_error()

    # Output results
    print("Monte Carlo took {0} s to run.".format(MC_exec_time))
    print("\nResults from Monte Carlo:\n")
    print("Input Covariance:\n{0}".format(situ.P_xx_uu))
    print("Pff:\n{0}".format(situ.P_ff_MC))
    print("Pmm:\n{0}".format(situ.P_mm_MC))
    print("\nLinear Covariance took {0} s to run.".format(LC_exec_time))
    print("\nResults from Linear Covariance:\n")
    print("Qx:\n{0}".format(situ.Q_x))
    print("Qu:\n{0}".format(situ.Q_u))
    print("Rx:\n{0}".format(situ.R_x))
    print("Ru:\n{0}".format(situ.R_u))
    print("\nPff:\n{0}".format(situ.P_ff_LC))
    print("Pmm:\n{0}".format(situ.P_mm_LC))
    print("\nDifferences:\n")
    print("Pff:\n{0}".format(situ.P_ff_MC-situ.P_ff_LC))
    print("Pmm:\n{0}".format(situ.P_mm_MC-situ.P_mm_LC))
    print("\nErrors:\n")
    print("Pff:\n{0}".format(situ.P_ff_error))
    print("Pmm:\n{0}".format(situ.P_mm_error))