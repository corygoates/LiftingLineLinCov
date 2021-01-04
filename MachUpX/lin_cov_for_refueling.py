"""Analysis methods for aircraft tailing scenario."""

import json
import os
import copy
import random
import time

import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import multiprocessing as mp
import subprocess as sp
import scipy.stats as stats
import machupX as mx


class TailingScenario:
    """Models one aircraft tailing another aircraft using MachUpX. Contains
    methods for linear covariance and Monte-Carlo analysis.

    Parameters
    ----------
    scene_input : dict or str
        Scene input for MachUpX. Should contain aircraft.

    lead : str
        Name of the lead aircraft.

    tail : str
        Name of the tailing aircraft.

    trim_iterations : int, optional
        Number of times to trim each aircraft. Defaults to 2.
    """


    def __init__(self, **kwargs):

        # Load scene
        self.scene = mx.Scene(scene_input=kwargs["scene_input"])

        # Store names
        self._lead = kwargs["lead"]
        self._tail = kwargs["tail"]

        # Trim
        self._trim_aircraft(kwargs.get('trim_iterations', 2))


    def _trim_aircraft(self, N):
        # Trims the aircraft at their current positions and velocities

        # Iteratively trim each
        for _ in range(N):
            self.scene.pitch_trim_using_orientation(aircraft=self._lead, verbose=True)
            self.scene.pitch_trim_using_orientation(aircraft=self._tail, verbose=True)


    def export_stl(self, **kwargs):
        """Exports an stl of the scene."""
        self.scene.export_stl(**kwargs)


    def get_derivatives(self):
        """Determines the state derivatives of the tailing aircraft for LinCov analysis."""

        # Get derivatives
        state_derivs = self.scene.state_derivatives(aircraft=self._tail)
        print(json.dumps(state_derivs, indent=4))
        control_derivs = self.scene.control_derivatives(aircraft=self._tail)
        print(json.dumps(control_derivs, indent=4))


    def calc_linear_uncertainties(self, state_unc, control_unc):
        """Determines the force and moment uncertainties based on the given state and control
        uncertainties using linear covariance analysis.

        Parameters
        ----------
        state_unc : list
            Uncertainties in the 13-element state vector.

        control_unc : dict
            Uncertainties in the control deflections.

        Returns
        -------
        P_ff : ndarray
            Force uncertainties.

        P_mm : ndarray
            Moment uncertainties.
        """

        # Get derivative matrices
        self.get_derivatives()


if __name__=="__main__":

    # Initialize scenario
    scene_input = "scene.json"
    scenario = TailingScenario(scene_input=scene_input, lead='tanker', tail='drone', trim_iterations=4)

    # Export stl
    scenario.export_stl(filename='tailing_scene.stl')
    
    # Initialize uncertainties
    state_unc = np.zeros(13)
    control_unc = np.zeros(2)

    # Propagate uncertainties using linear covariance
    P_ff_LC, P_mm_LC = scenario.calc_linear_uncertainties(state_unc, control_unc)

    # Propagate uncertainties using Monte-Carlo

    # Compare
