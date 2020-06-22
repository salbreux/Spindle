#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 17:14:42 2019

@author: salbreg
"""


import matplotlib
matplotlib.use('agg')
import numpy as np
from MonopolarFit import monopolar_fit


# =============================================================================
# SET PARAMETERS
# =============================================================================
#other parameters are set in Monopolar.py
TAU = 11.5
K_OFF_FAR = 0.011

print("TAU="+str(TAU)+' ')
print("K_OFF_FAR="+str(K_OFF_FAR)+' ')

# =============================================================================
# CALCULATE ERROR EXPLICITELY
# =============================================================================
this_fit = monopolar_fit()
#initial conditions for unknown factors controlling the 
#ratio between the spatiotemporal mean of the simulated concentration
#and the spatiotemporal mean of the data.
ALPHA_VECT = np.ones(12)
#run 8 iterations where ALPHA is reestimated
for k in range(8):
    print('run '+str(k)+' ')
    error_vect = []
    final_mean_vect = []
    #calculate the error for each cell by comparing simulation and experiment.
    #numbering of cells starts from 1 and ends at 12
    for cell_number in range(1, 13):
        error_vect.append([this_fit.error_for_cell(cell_number,
                                                   ALPHA_VECT[cell_number-1],
                                                   TAU,
                                                   K_OFF_FAR)])
        #record the final value of the mean concentration- to be used as a guess for alpha
        final_mean_vect.append(np.mean(this_fit.this_simulation.sol_c[:, :]))    
    print('choices of alpha:\n')  
    print(ALPHA_VECT)
    print('error:\n')  
    print(error_vect)
    print('\n overall error: '+str(np.linalg.norm(error_vect)))
    ALPHA_VECT = final_mean_vect