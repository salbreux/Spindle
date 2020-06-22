#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 14:19:19 2019

@author: salbreg
"""
import numpy as np
from scipy.integrate import odeint
#from scipy.fftpack import diff as psdiff
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#----------------------------------------------------------
# Class Monopolar
# a class that contains the tools to simulate the 1D equation
#----------------------------------------------------------
class monopolar(object):
    """this class contains methods to run a 1D monopolar spatiotemporal simulation"""
    
    #length scale of nucleus inhibition - given in um
    nucleus_inhibition_range = 11.0
    #distance between centrosome and nucleus - given in um
    centrosome_nucleus_distance = -4.9
    #characteristic velocity of the nucleus position - given in um/min
    V_0 = 6.7
    # parameters quantifying the distribution of microtubules
    MT_MU = np.log(10.28)
    MT_SIGMA = np.log(1.26)
    #diffusion constant of LGN - given in um^2/min
    D = 0.01 
    # on and off rates, away and close to the nucleus -given in min^(-1)
    k_on_far = 1.0 / 11.5
    k_on_near = 1.0 / 11.5
    k_off_far = 0.011
    k_off_near = 1.0 / 11.5
    
    #arrays are used to store the spatially dependent on and off rates
    k_on_array = []
    k_off_array = []
    #contain the solution
    sol = []
    
    def __init__(self, nb_x_points_value, L_value, total_time_value, nb_time_points_value):
        # Domain size
        self.L = L_value
        #number of points in the domain
        self.nb_x_points = nb_x_points_value
        # Set dx interval
        self.dx = float(self.L) / float(self.nb_x_points)
        # Vector of x positions
        self.x_array = np.linspace(0, (1.0 - 1.0 / float(self.nb_x_points)) * self.L, self.nb_x_points)
        # position of the nucleus - by default positioned in the middle of the domain
        self.nucleus_position = self.L / 2.0 
         # position of the nucleus within the periodic domain 
        self.nucleus_position_periodic = self.nucleus_position % self.L
        # position of the centrosome
        self.centrosome_position = self.nucleus_position + self.centrosome_nucleus_distance
        # position of the centrosome within the periodic domain
        self.centrosome_position_periodic = self.centrosome_position % self.L
        # Set the initial conditions.
        self.c_init = np.zeros(self.nb_x_points)
        # Set the time sample grid.
        self.total_time = total_time_value
        self.nb_time_points = nb_time_points_value
        self.time_array = np.linspace(0, self.total_time, self.nb_time_points)
        
        # Obtain profiles of kon and koff
        self.set_kon_koff_nMT()
        
# =============================================================================
# FUNCTIONS TO CALCULATE THE SOLUTION WITH NUCLEUS MOTION
# =============================================================================
    def set_kon_koff_nMT(self):
        """ function that calculates the profile of kon, koff, and microtubule density nMT
        """
        distance_array = periodic_distance(self.x_array, self.nucleus_position_periodic, self.L)
        self.k_on_array = (self.k_on_far * (distance_array >  self.nucleus_inhibition_range)
                          + self.k_on_near * (distance_array <=  self.nucleus_inhibition_range) )
        self.k_off_array = (self.k_off_far * (distance_array >  self.nucleus_inhibition_range)
                          + self.k_off_near * (distance_array <=  self.nucleus_inhibition_range) )
        #for the distribution of MT, use the addition of two boxes before and after the main box
        self.nMT = (signed_log_normal(self.x_array-self.centrosome_position_periodic,
                                      self.MT_MU,
                                      self.MT_SIGMA) 
                    +   signed_log_normal(self.x_array-self.centrosome_position_periodic - self.L,
                                          self.MT_MU,
                                          self.MT_SIGMA) 
                    +   signed_log_normal(self.x_array-self.centrosome_position_periodic + self.L,
                                          self.MT_MU,
                                          self.MT_SIGMA)  )
    def dcdt_dxndt(self, cx, t, L):
        """Differential equations for the concentration equation, discretized in x,
        and for the evolution of the nucleus position equation.
        """
        #cx contains the concentration vector and the nucleus position,
        #so first extracts the concentration vector only
        c_array=cx[0:-1]
        #then update the position of the nucleus
        self.nucleus_position = cx[-1]
        # update the position of the nucleus within the periodic domain
        self.nucleus_position_periodic = self.nucleus_position % self.L
        #then update the position of the centrosome
        self.centrosome_position = self.nucleus_position + self.centrosome_nucleus_distance
        # position of the centrosome within the periodic domain
        self.centrosome_position_periodic = self.centrosome_position % self.L
        # Calculate the profile of kon/koff
        self.set_kon_koff_nMT()
        #compute the x derivatives
        cxx = np.zeros_like(c_array)
        cxx[1:-1] = (c_array[2:] + c_array[0:-2] - 2.0 * c_array[1:-1]) / (self.dx **2)
        cxx[0] = (c_array[1] + c_array[-1] - 2.0 * c_array[0]) / (self.dx **2)
        cxx[-1] = (c_array[-2] + c_array[0] - 2.0 * c_array[-1]) / (self.dx **2)
        # Compute dc/dt.    
        dcdt_vector =  self.D * cxx + self.k_on_array - self.k_off_array * c_array
        # Compute dx/dt - use 3 boxes to calculate the integral with the distribution of MTs        
        dxndt = self.V_0 * self.dx * np.sum(c_array * self.nMT)
        return np.concatenate((dcdt_vector, np.array([dxndt])))
    def solve_equations(self):
        """Use odeint to solve the concentration equation on a periodic domain.    
        """
        #the second argument of odeint is the initial condition
        initial_condition = np.concatenate((self.c_init, np.array([self.nucleus_position])))
        sol = odeint(self.dcdt_dxndt,
                          initial_condition,
                          self.time_array,
                          args=(self.L,),
                          mxstep=500000)
        self.sol_c = np.delete(sol, -1, axis=1)
        self.nucleus_position_vect = sol[:, -1]
        return [self.sol_c, self.nucleus_position_vect]
# =============================================================================
# FUNCTIONS TO CALCULATE THE SOLUTION WITH IMPOSED NUCLEUS MOTION
# ============================================================================= 
    def set_kon_koff(self):
        """ function that calculates the profile of kon, koff
        """
        distance_array = periodic_distance(self.x_array, self.nucleus_position_periodic, self.L)
        self.k_on_array = (self.k_on_far * (distance_array >  self.nucleus_inhibition_range)
                          + self.k_on_near * (distance_array <=  self.nucleus_inhibition_range) )
        self.k_off_array = (self.k_off_far * (distance_array >  self.nucleus_inhibition_range)
                          + self.k_off_near * (distance_array <=  self.nucleus_inhibition_range) ) 
    def dcdt_imposed_nucleus(self, c_array, t, L, nucleus_position_function):
        """Differential equations for the concentration equation, discretized in x,
        and for the evolution of the nucleus position equation.
        """
        #cx contains the concentration vector and the nucleus position,
        #so first extracts the concentration vector only
        #then update the position of the nucleus
        self.nucleus_position = nucleus_position_function(t)
        # update the position of the nucleus within the periodic domain
        self.nucleus_position_periodic = self.nucleus_position % self.L
        # Calculate the profile of kon/koff
        self.set_kon_koff()
        #calculate the second derivative using spline interpolation
        x_array_extended=np.concatenate((self.x_array, [self.x_array[0]+self.L]))
        c_array_extended=np.concatenate((c_array, [c_array[0]]))
        c_spline = CubicSpline(x_array_extended, c_array_extended, bc_type='periodic')
        #compute the x derivatives with a spline
        cxx=c_spline(self.x_array,2)
        # Compute dc/dt.    
        dcdt_vector =  self.D * cxx + self.k_on_array - self.k_off_array * c_array
        return dcdt_vector
    def solve_equations_imposed_nucleus(self, nucleus_position_function):
        """ solve equations for concentration with
        a sequence of nucleus position that is imposed
        """
        sol = odeint(self.dcdt_imposed_nucleus,
                          self.c_init,
                          self.time_array,
                          args=(self.L, nucleus_position_function),
                          mxstep=500000)
        self.sol_c = sol
        self.nucleus_position_vect = [nucleus_position_function(t) for t in self.time_array]
        return self.sol_c
# =============================================================================
#     FUNCTIONS TO PLOT THE SOLUTION
# =============================================================================
    def plot_c_profile(self):
        """ plot subfigures with concentration profiles and nucleus position over time"""
        fig = plt.figure(1)
        plt.clf()
        ax = fig.subplots(nrows=2, ncols=2)
        plt.subplots_adjust(hspace=0.5, wspace=0.5)
        ax[0,0].set_title('concentration profile at different times',fontsize='medium')
        ax[0,0].set_xlabel('x position (um',fontsize='small')
        ax[0,0].set_ylabel('normalized concentration',fontsize='small')
        for c_vect in self.sol_c:
            ax[0,0].plot(self.x_array,c_vect)
        ax[0,1].set_title('nucleus position',fontsize='medium')
        ax[0,1].set_xlabel('time (min)',fontsize='small')
        ax[0,1].set_ylabel('nucleus position (um)',fontsize='small')
        ax[0,1].plot(self.time_array, self.nucleus_position_vect)
        nucleus_velocity = np.diff(self.nucleus_position_vect) / np.diff(self.time_array)
        ax[1,0].plot(self.time_array[0:-1], nucleus_velocity)
        ax[1,0].set_title('nucleus velocity',fontsize='medium')
        ax[1,0].set_xlabel('time in minutes',fontsize='small')
        ax[1,0].set_ylabel('velocity (um/min)',fontsize='small')
        ax[1,1].set_title('final profile of MT distribution',fontsize='medium')
        ax[1,1].set_xlabel('x position (um)',fontsize='small')
        ax[1,1].set_ylabel('normalized signed distribution',fontsize='small')
        ax[1,1].plot(self.x_array, self.nMT)
        plt.show()
    def plot_kymograph(self):
        """ plot kymograph"""
        fig = plt.figure(2)
        plt.clf()
        ax = fig.subplots()
        max_concentration=np.max(np.abs(self.sol_c))
        kymograph_plot=ax.imshow(np.transpose(self.sol_c),
                                 clim=[0, max_concentration],
                                 aspect='auto',
                                 extent=[0, self.total_time,0,self.L],
                                 interpolation='none')
        cbar = ax.figure.colorbar(kymograph_plot, ax=ax)
        cbar.ax.set_ylabel('concentration', rotation=-90, va="bottom")
        #ax.set_xticks(np.linspace(0,self.total_time,3))
        #ax.set_yticks(np.linspace(0,self.L,3))
        ax.set_xlabel('time (min)')
        ax.set_ylabel('x position (um)')
        ax.set_title('kymograph of concentration')
        plt.show()
    def animate(self):
        """trigger an animation to show the evolution of the concentration profile"""
        self.fig = plt.figure(3)
        plt.clf()
        x_range = [self.x_array[0], self.x_array[-1]]
        y_range = [np.min(self.sol_c), 1.5*np.max(self.sol_c)]
        self.ax = plt.axes()
        self.ax.set_xlim(x_range)
        self.ax.set_ylim(y_range)
        self.scat_conc_profile = plt.scatter(self.x_array,
                                             self.sol_c[0],
                                             color='b',
                                             animated=True)
        plt.xlabel('distance (um)')
        plt.ylabel('normalized concentration')
        plt.show()
        self.ani = animation.FuncAnimation(self.fig,
                                   func=self.update,
                                   frames=np.arange(1, len(self.time_array)),
                                   init_func=None,
                                   repeat=True,
                                   interval=10,
                                   blit=True)
        plt.show()
    def update(self, i):
        """ update function for animation"""
        self.scat_conc_profile.set_offsets(np.transpose(np.stack([self.x_array,self.sol_c[i]])))
        return self.scat_conc_profile, 
    def plot_nucleus_velocity_and_force(self, this_time_list):
        """ plot the nucleus velocity and force over time, to check agreement
        this time can be adjusted to plot the concentration profile at a specific time point"""
        fig=plt.figure(4)
        plt.clf()
        ax = fig.subplots(nrows=2, ncols=2)
        plt.subplots_adjust(hspace=0.5, wspace=0.5)
        #plot the nucleus position
        ax[0, 0].set_title('nucleus position',fontsize='medium')
        ax[0, 0].set_xlabel('time (min)',fontsize='small')
        ax[0, 0].set_ylabel('nucleus position (um)',fontsize='small')
        ax[0, 0].plot(self.time_array, self.nucleus_position_vect)
        #calculate the velocity
        nucleus_velocity = np.diff(self.nucleus_position_vect) / np.diff(self.time_array)
        #calculate the force at different times
        x_3boxes = np.concatenate((self.x_array - self.L, self.x_array, self.x_array + self.L))
        force_array = []
        for this_time_index_2 in range(self.time_array.shape[0]):
            nucleus_position = self.nucleus_position_vect[this_time_index_2]
            centrosome_position = nucleus_position + self.centrosome_nucleus_distance
            # position of the centrosome within the periodic domain
            centrosome_position_periodic = centrosome_position % self.L
            #recalculate the density of MTs at a specific time point
            nMT = signed_log_normal(x_3boxes - centrosome_position_periodic, self.MT_MU, self.MT_SIGMA)  
            c_array = np.array(self.sol_c[this_time_index_2])
            c_3boxes = np.concatenate((c_array, c_array, c_array))
            force_array.append(self.V_0 * self.dx * np.sum(c_3boxes * nMT))
        #plot the force and the velocity
        ax[0, 1].plot(self.time_array[0:-1], nucleus_velocity)
        ax[0, 1].set_title('nucleus velocity (blue) and force(red)',fontsize='medium')
        ax[0, 1].set_xlabel('time in minutes',fontsize='small')
        ax[0, 1].set_ylabel('velocity (um/min)',fontsize='small')
        ax[0, 1].plot(self.time_array, force_array, color='red')
        #plot concentration profiles at the selected time points
        string_times = ' '
        for this_time_index in this_time_list:
            string_times = string_times + str(round(self.time_array[this_time_index], 2)) + ' '
        for this_time_index in this_time_list:
            c_array = self.sol_c[this_time_index]
            ax[1, 0].plot(self.x_array, c_array)
            ax[1, 0].set_title('concentration profile at '+string_times+' mins',fontsize='medium')
            ax[1, 0].set_xlabel('position (um)',fontsize='small')
            ax[1, 0].set_ylabel('concentration',fontsize='small')
        #plot concentration profiles and nMT distribution in 3 boxes
        for this_time_index in this_time_list:
            #plot one particular time point, copied concentration profile
            c_3boxes = np.concatenate((c_array, c_array, c_array))
            nucleus_position = self.nucleus_position_vect[this_time_index]
            centrosome_position = nucleus_position + self.centrosome_nucleus_distance
            # position of the centrosome within the periodic domain
            centrosome_position_periodic = centrosome_position % self.L
            #recalculate the density of MTs at a specific time point
            nMT = signed_log_normal(x_3boxes - centrosome_position_periodic, self.MT_MU, self.MT_SIGMA) 
            ax[1, 1].plot(x_3boxes, c_3boxes)
            ax[1, 1].plot(x_3boxes, nMT, color='red')
        ax[1, 1].set_title('concentration profile at '+string_times+' mins;\n red:nMT',fontsize='medium')
        ax[1, 1].set_xlabel('position (um)',fontsize='small')
        ax[1, 1].set_ylabel('concentration',fontsize='small')
        plt.show()
# =============================================================================
#  ADDITIONAL USEFUL FUNCTIONS
# =============================================================================
def periodic_distance(x_array, x_position, L):
    """ return a vector of distances to x_positions for points in x_array, assuming
    periodic boundary conditions in a domain between 0 and L
    """
    distance_1 = np.abs(x_array - x_position)
    distance_2 = np.abs(x_array - x_position - L)
    distance_3 = np.abs(x_array - x_position + L)
    distance = np.array([min(l) for l in zip(distance_1, distance_2, distance_3)])
    return distance

def signed_log_normal(x, mu, sigma):
    """return a variable with log-normal distribution"""
    #some manipulation is required to avoid evaluating the log in 0
    result = np.zeros_like(x)
    mask = (x!=0)
    nonzero_x = x[mask]
    result[mask] =  0.5 * (1.0 /
                     (nonzero_x * sigma * np.sqrt(2.0 * np.pi))
                     * np.exp(-(np.log(abs(nonzero_x)) - mu) ** 2 / 2.0 / (sigma ** 2)))
    return result



