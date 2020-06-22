#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 15:23:59 2019

@author: salbreg
"""

import os
import os.path
import csv
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from Monopolar import monopolar

#----------------------------------------------------------
# Class MonopolarFit
# a class that contains the tools to fit experimental data
#----------------------------------------------------------
class monopolar_fit(object):
    """this class contains method to compare monopolar simulations to experimental data"""
    #experimentally measured values of DNA-centrosome distance
    spindle_length = 4.9
    def __init__(self):
        self.this_simulation = None
        self.total_time = 0
        self.L = 0
    def error_for_cell(self, cell_number, ALPHA, tau, k_off_far):
        """ calculate error between prediction and experiment
        ALPHA is a multiplicative factor for initial condition"""
        #first extract useful information from the experimental data
        [nucleus_position_function,
         NB_POINTS,
         NB_TIME_POINTS,
         c_array_init_data,
         lgn_normalized_data] = self.extract_data_for_cell(cell_number)
        #create a simulation object
        self.this_simulation = monopolar(NB_POINTS,
                                         self.L,
                                         self.total_time,
                                         NB_TIME_POINTS)
        #set initial condition
        self.this_simulation.c_init = ALPHA * c_array_init_data
        #set parameters
        self.this_simulation.k_on_far = 1.0 / tau
        self.this_simulation.k_on_near = 1.0 / tau
        self.this_simulation.k_off_far = k_off_far
        self.this_simulation.k_off_near = 1.0 / tau
        #solve the differential equation
        self.this_simulation.solve_equations_imposed_nucleus(nucleus_position_function)
        self.plot_kymographs(self.this_simulation.sol_c/np.mean(self.this_simulation.sol_c), lgn_normalized_data)
        return np.linalg.norm(self.this_simulation.sol_c/np.mean(self.this_simulation.sol_c)-lgn_normalized_data)
    def save_solution(self, file_name):
        with open(file_name, mode='wb') as file_to_save:
            file_writer = csv.writer(file_to_save, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for this_row in self.this_simulation.sol_c:
                file_writer.writerow(this_row)
        file_to_save.close()
    def extract_data_for_cell(self, cell_number):
        """ function which extracts quantities that are useful for simulations
        from experimental data
        """
        #load experimental data for LGN kymograph
        current_directory = os.getcwd()
        file_path = os.path.join(current_directory,
                                 'Data_Monopolar/Cell'
                                 +str(cell_number)
                                 +'/Cell'
                                 +str(cell_number)
                                 +'_conc.dat')
        lgn_data = np.loadtxt(file_path, delimiter=',')
        lgn_normalized_data = lgn_data / np.mean(lgn_data)
        #load data for initial condition of LGN concentration
        current_directory = os.getcwd()
        file_path = os.path.join(current_directory,
                                 'Data_Monopolar/Cell'
                                 +str(cell_number)
                                 +'/Cell'
                                 +str(cell_number)
                                 +'_lgn_init.dat')
        lgn_init_data = np.loadtxt(file_path, delimiter=',')
        x_array_data = lgn_init_data[:, 0]
        #obtain an initial condition by dividing by the average
        c_array_init_data = lgn_init_data[:, 1] / np.mean(lgn_data)

        #extract spatial information from initial array of concentration
        DELTA_X = lgn_init_data[1, 0] -  lgn_init_data[0, 0]
        NB_POINTS = len(x_array_data)
        L_VALUE = DELTA_X * NB_POINTS
        #load data for centrosome position
        file_path = os.path.join(current_directory,
                                 'Data_Monopolar/Cell'
                                 +str(cell_number)
                                 +'/Cell'
                                 +str(cell_number)
                                 +'_Mech.dat')
        mech_data = np.loadtxt(file_path, delimiter=',')
        time_array_data, centrosome_position_data, _, spindle_angle_data = np.transpose(mech_data)
        #extract a non periodic version of the centrosome position
        #does this by assuming that displacements larger
        #than L/2 correspond to a jump across the periodic boundary
        centrosome_displacement = [correct_displacement(this_displacement, L_VALUE)
                                   for this_displacement
                                   in np.diff(centrosome_position_data)]
        centrosome_non_periodic_data = np.concatenate(([centrosome_position_data[0]],
                                                       centrosome_position_data[0]
                                                       + np.cumsum(centrosome_displacement)))
        nucleus_position_data = (centrosome_non_periodic_data
                                 + self.spindle_length
                                 * np.cos(np.pi / 180.0 * spindle_angle_data))
        # extrapolation is introduced here for calls of the ODE solver
        # that go beyond the final time of experiment
        # which is also the final requested time to the solver
        # this should not influence the final result
        nucleus_position_function = interp1d(time_array_data,
                                             nucleus_position_data,
                                             fill_value="extrapolate")
        self.total_time = time_array_data[-1] - time_array_data[0]
        self.L = L_VALUE
        NB_TIME_POINTS = len(time_array_data)
        return [nucleus_position_function,
                NB_POINTS,
                NB_TIME_POINTS,
                c_array_init_data,
                lgn_normalized_data]
    def error_for_all_cells(self, ALPHA_vect, tau, k_off_far):
        """ calculate the error for the 12 cells """
        error_vect = []
        for cell_number in range(1, 13):
            error_vect.append([self.error_for_cell(cell_number, ALPHA_vect[cell_number-1], tau, k_off_far)])
        return error_vect
            
# =============================================================================
#         PLOTTING FUNCTIONS
# =============================================================================
    def plot_kymographs(self, array_1, array_2):
        """ plot kymograph"""
        fig = plt.figure(2)
        plt.clf()
        ax = fig.subplots(nrows=2, ncols=1)
        plt.subplots_adjust(hspace=0.5, wspace=0.5)
        max_concentration_1 = np.max(np.abs(array_1))
        max_concentration_2 = np.max(np.abs(array_2))
        max_concentration = np.max([max_concentration_1, max_concentration_2])
        kymograph_plot=ax[0].imshow(np.transpose(array_1),
                                 clim=[0, max_concentration],
                                 aspect='auto',
                                 extent=[0, self.total_time, 0, self.L],
                                 interpolation='none')
        cbar = ax[0].figure.colorbar(kymograph_plot, ax=ax)
        cbar.ax.set_ylabel('concentration', rotation=-90, va="bottom")
        ax[0].set_xlabel('time (min)')
        ax[0].set_ylabel('x position (um)')
        ax[0].set_title('kymograph of concentration')
        kymograph_plot=ax[1].imshow(np.transpose(array_2),
                                 clim=[0, max_concentration],
                                 aspect='auto',
                                 extent=[0, self.total_time,0,self.L],
                                 interpolation='none')
        cbar = ax[1].figure.colorbar(kymograph_plot, ax=ax)
        cbar.ax.set_ylabel('concentration', rotation=-90, va="bottom")
        ax[1].set_xlabel('time (min)')
        ax[1].set_ylabel('x position (um)')
        ax[1].set_title('kymograph of concentration')
        plt.show()
# =============================================================================
# ADDITIONAL USEFUL FUNCTIONS
# =============================================================================

def correct_displacement(number, L_VALUE):
    """
    correct displacement by assuming that displacements larger
    than L/2 correspond to a jump across the periodic boundary 
     """
    if number > L_VALUE / 2.0:
        return number - L_VALUE
    elif number < - L_VALUE / 2.0:
        return number + L_VALUE
    return number
