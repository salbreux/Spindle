#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 15:55:04 2019

@author: salbreg
"""

import os
import csv
import os.path
import time
from Monopolar import monopolar

# =============================================================================
# %% RUN A SIMULATION
# =============================================================================
#set simulation parameters
#(other default parameters are set in Monopolar.py)
NB_POINTS = 400
L_VALUE = 120.0
TOTAL_TIME = 140
NB_TIME_POINTS = 2000
#run simulation 
this_simulation=monopolar(NB_POINTS,
                          L_VALUE,
                          TOTAL_TIME,
                          NB_TIME_POINTS)
start = time.time()
sol = this_simulation.solve_equations()
end = time.time()
print('execution time', end-start)
#plot results
this_simulation.plot_c_profile()
this_simulation.plot_kymograph()
this_simulation.plot_nucleus_velocity_and_force([0, 1000])
this_simulation.animate()

# =============================================================================
# %% SAVE SOLUTION
# =============================================================================
sol_concentration = sol[0]
sol_nucleus = sol[1]
if not os.path.exists('results'):
    os.mkdir('results')
folder_name='./results'
csvfile = folder_name + "/result_concentration.csv"
with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(sol_concentration)  
csvfile_2 = folder_name + "/result_nucleus_position.csv"
with open(csvfile_2, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerow(sol_nucleus)
    
    
