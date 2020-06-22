#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 22:16:10 2019

@author: salbreg
"""

import os
import os.path
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Bipolar import bipolar

# =============================================================================
# %% BASIC PARAMETER DEFINITIONS
# =============================================================================
NB_POINTS = 800
ELLIPSE_L = 18 #this will not be used here
ELLIPSE_e = 0.1 #this will not be used here
TOTAL_TIME = 30
NB_TIME_POINTS = 901
TAU_R = 12
LENGTH_MTs = 15
NUCLEUS_INHIBITION_RANGE = 4
CENTROSOME_NUCLEUS_DISTANCE = 5.6
D_VALUE = 0.01
K_ON_FAR = 1.0 / 11.5
K_ON_NEAR = 1.0 / 11.5
K_OFF_FAR = 0.011
K_OFF_NEAR = 1.0 / 11.5
METAPHASE_PLATE_LENGTH = 14
EQUILIBRIUM_CONCENTRATION_FAR = K_ON_FAR / K_OFF_FAR
EQUILIBRIUM_CONCENTRATION_NEAR = K_ON_NEAR / K_OFF_NEAR
SPINDLE_INITIAL_ANGLE = np.pi / 4.0
#parameters for cell shape as a function of time
LFINAL = 10.5
LM0 = 27.3
Lm0 = 7.1
tM0 = -3.1
tm0 = 4.2
TAUM = 6.2
TAUm = 12.0
#time at which the spindle starts to rotate
SPINDLE_ROTATION_TIME = 9
#function that controls the shape change
#there is a division by e which could be problematic if e was too small
#but with the parameters chosen here it does not get smaller than 0.1.
def get_new_shape(t):
    """return cell shape parameters and spindle mobility according to time
    t is the time
    """
    #ellipse semi-major axis
    ellipse_LM = LFINAL + (LM0 - LFINAL) * (1 - np.tanh((t-tM0) / TAUM)) / 2.0
    #ellipse semi-minor axis
    ellipse_Lm = LFINAL + (Lm0 - LFINAL) * (1 - np.tanh((t-tm0) / TAUm)) / 2.0
    #calculate eccentricity
    ellipse_e = np.sqrt(1.0 - np.square(ellipse_Lm / ellipse_LM))
    ellipse_L = ellipse_Lm
   #time derivatives
    dt_ellipse_LM = ((LM0 - LFINAL) * 1.0 / TAUM 
                     * (- 1.0/ np.square(np.cosh((t-tM0) / TAUM))) / 2.0)
    dt_ellipse_Lm = ((Lm0 - LFINAL) * 1.0 / TAUm
                     * (- 1.0/ np.square(np.cosh((t-tm0) / TAUm))) / 2.0)
    dt_ellipse_e = (1.0 / ellipse_e * np.square(ellipse_Lm/ellipse_LM) 
                    * (dt_ellipse_LM / ellipse_LM 
                       - dt_ellipse_Lm / ellipse_Lm))
    return [ellipse_L,
            ellipse_e,
            dt_ellipse_Lm,
            dt_ellipse_e] 


# =============================================================================
# %% RUN SIMULATION
# =============================================================================
this_simulation = bipolar(NB_POINTS,
                          ELLIPSE_L,
                          ELLIPSE_e,
                          TOTAL_TIME,
                          NB_TIME_POINTS,
                          change_shape=1,
                          get_new_shape=get_new_shape)
#set-up parameters
this_simulation.spindle_angle = SPINDLE_INITIAL_ANGLE 
this_simulation.tau_r = TAU_R
this_simulation.length_MTs = LENGTH_MTs
this_simulation.nucleus_inhibition_range = NUCLEUS_INHIBITION_RANGE
this_simulation.centrosome_nucleus_distance = CENTROSOME_NUCLEUS_DISTANCE
this_simulation.D = D_VALUE
this_simulation.k_on_far = K_ON_FAR
this_simulation.k_on_near = K_ON_NEAR
this_simulation.k_off_far = K_OFF_FAR
this_simulation.k_off_near = K_OFF_NEAR
this_simulation.metaphase_plate_length = METAPHASE_PLATE_LENGTH
# start the simulation from the low value of concentration of LGN
this_simulation.c_init = EQUILIBRIUM_CONCENTRATION_NEAR * np.ones(NB_POINTS)
#run simulation, recording the time necessary
start = time.time()
sol = this_simulation.solve_equations_two_steps(SPINDLE_ROTATION_TIME, TAU_R)
end = time.time()
print('execution time', end-start)

# =============================================================================
# %% PLOT THE RESULT
# =============================================================================
#first recalculate geometric quantities for plotting
start = time.time()
this_simulation.calculate_geometry_from_solution()
end = time.time()
print('execution time', end-start)
#plot a kymograph of concentration
this_simulation.plot_kymograph()
#show an animmation of spindle motion, cell shape change and cortical LGN
this_simulation.animate_cell()
# =============================================================================
# %% PLOT AND SAVE A NUMBER OF ADDITIONAL GRAPHS
# =============================================================================
# save a few still frames
if not os.path.exists('results'):
    os.mkdir('results')
folder_name='./results'
fig_file = folder_name + "/simulation_cell_successive_times.eps"
this_simulation.plot_cell_sequence([0, 150, 300, 450, 600])
plt.savefig(fig_file)
# =============================================================================
# SAVE THE CONCENTRATION PROFILE SOLUTION
# =============================================================================
#save data for spindle angle as a csv file

csvfile = folder_name + "/result_c.csv"
with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(this_simulation.sol_c)   
# =============================================================================
# %% SAVE THE ANGLE AS A FUNCTION OF TIME SOLUTION
# =============================================================================
#save data for spindle angle as a csv file
csvfile = folder_name + "/result_angle.csv"
with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerow(this_simulation.spindle_angle_vect) 
csvfile = folder_name + "/result_time_array.csv"
with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerow(this_simulation.time_array) 
# =============================================================================
# %% SAVE A MOVIE OF THE SIMULATION
# =============================================================================
 # Set up formatting for the movie files
plt.figure(3)
plt.clf()
this_simulation.animate_cell()
WRITER = animation.writers['ffmpeg']
folder_name = './results'
MOVIE_WRITER = WRITER(codec='mpeg4', bitrate=1e6, fps=24)
movie_file = folder_name + "/result_movie.mp4"
this_simulation.ani.save(movie_file, writer=MOVIE_WRITER, dpi=400)