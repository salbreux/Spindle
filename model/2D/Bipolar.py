#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 14:19:19 2019

@author: salbreg
"""
import numpy as np
import scipy as sp
from scipy.integrate import odeint
#from scipy.fftpack import diff as psdiff
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse
from matplotlib.patches import Rectangle
import matplotlib.lines as mlines
from matplotlib.collections import LineCollection

#----------------------------------------------------------
# Class bipolar
# a class that contains the tools to simulate the 2D equations
# for a bipolar spindle driven by LGN
#----------------------------------------------------------
class bipolar(object):
    """this class contains methods to run
    a 2D bipolar spindle spatiotemporal simulation"""
    #length scale of nucleus inhibition - given in um
    nucleus_inhibition_range = 4.0
    #distance between centrosome and nucleus - given in um
    centrosome_nucleus_distance = 5.6
    #length of the metaphase plate - given in um
    metaphase_plate_length = 14.0
    #maximal microtubule length - given in um
    length_MTs = 15.0
    #characteristic time scale of spindle rotation - given in min
    tau_r = 12.0
    #diffusion constant of LGN - given in um^2/min
    D = 0.01
    # on and off rates, away and close to the nucleus - given in min^(-1)
    k_on_far = 1.0 / 11.5
    k_on_near = 1.0 / 11.5
    k_off_far = 0.011
    k_off_near = 1.0 / 11.5
    #resolution of initial rough search for crossing points of MT
    #with cell contour
    #if MT crossing points are closer than 1/EXTREMAL_POINTS_SEARCH_RES
    #the crossing points might be missed
    EXTREMAL_POINTS_SEARCH_RES = 100
    #arrays are used to store the spatially dependent on and off rates
    k_on_array = []
    k_off_array = []
    #these vectors will be used to store the solution
    k_on_array_vect = []
    k_off_array_vect = []
    centrosome_position_1_vect = []
    centrosome_position_2_vect = []
    metaphase_plate_1_vect = []
    metaphase_plate_2_vect = []
    circular_metaphase_plate_vect = []
    ellipse_e_vect = []
    ellipse_L_vect = []
    tau_r_vect = []
    torque_vect = []
    spindle_angle_vect = []
    #for the case where there are two steps in the solution
    spindle_angle_vect_1 = []
    spindle_angle_vect_2 = []
    sol_c_1 = []
    sol_c_2 = []
    # whether to update the shape - by default set to False
    change_shape = False
    # whether to consider the metaphase plate as a circle
    #for LGN inhibition - by default set to False
    circular_metaphase_plate = False
    #variables used for animation
    ani = None
    fig = None
    ax = None
    scat_conc_profile = None
    animation_patches_cell_shape_contour_0_0 = None
    animation_patches_MT_1_patch_0_0 = None
    animation_patches_MT_2_patch_0_0 = None
    animation_patches_Spindle_patch_0_0 = None
    animation_patches_metaphase_plate_line_0_0 = None
    animation_patches_metaphase_plate_circle_0_0 = None
    animation_cell_shape_0_1 = None
    animation_patches_MT_1_patch_0_1 = None
    animation_patches_MT_2_patch_0_1 = None
    animation_patches_Spindle_patch_0_1 = None
    animation_patches_metaphase_plate_line_0_1 = None
    animation_patches_metaphase_plate_circle_0_1 = None
    animation_patches_lc = None
    animation_cell_shape_1_0 = None
    animation_patches_MT_1_patch_1_0 = None
    animation_patches_MT_2_patch_1_0 = None
    animation_patches_Spindle_patch_1_0 = None
    animation_patches_Ran_inhibition_patch_1_1_0 = None
    animation_patches_Ran_inhibition_patch_2_1_0 = None
    animation_patches_Ran_inhibition_patch_3_1_0 = None
    animation_patches_Ran_inhibition_patch_circular_1_0 = None
    animation_patches_metaphase_plate_line_1_0 = None
    animation_patches_metaphase_plate_circle_1_0 = None
    animation_patches_cell_shape_contour_1_0 = None
    animation_patches_Ran_inhibition_patch_1_1_1 = None
    animation_patches_Ran_inhibition_patch_3_1_1 = None
    animation_patches_Ran_inhibition_patch_2_1_1 = None
    animation_patches_Ran_inhibition_patch_circular_1_1 = None
    animation_patches_metaphase_plate_line_1_1 = None
    animation_patches_metaphase_plate_circle_1_1 = None
    animation_patches_cell_shape_contour_1_1 = None
    animation_patches_lc_1_1 = None
    def __init__(self,
                 nb_theta_points_value,
                 ellipse_L,
                 ellipse_e,
                 total_time_value,
                 nb_time_points_value,
                 **kwargs):
        """ initialize the function
        nb_theta_points_value is the number of points
        on the contour (polar coordinates are used)
        ellipse_L, ellipse_e  define the ellipse shape
        with r(theta) = L / sqrt(1 - e^2 cos(theta)^2)
        total_time_value is the total time to simulate
        nb_time_points_value the time resolution
        """
        #number of points in the domain
        self.nb_theta_points = nb_theta_points_value
        #resolution of search of crossing points of MTs
        # is set to number of points of domain
        self.EXTREMAL_POINTS_SEARCH_RES = nb_theta_points_value + 1
        # Set dtheta interval
        self.dtheta = 2.0 * np.pi / float(self.nb_theta_points)
        # Vector of theta positions
        self.theta_array = np.linspace(0.0,
                                       (1.0
                                        - 1.0 / float(self.nb_theta_points))
                                       * 2.0
                                       * np.pi,
                                       self.nb_theta_points)
        # geometrical quantities for the ellipse representing cell shape
        self.ellipse_L = ellipse_L
        self.ellipse_e = ellipse_e
        # angle of the spindle - by default 0
        self.spindle_angle = 0
        # position of the centrosomes and metaphase plate
        self.update_centrosome_metaphase_plate()
        # Set the initial conditions.
        self.c_init = np.zeros(self.nb_theta_points)
        # Set the time sample grid.
        self.total_time = total_time_value
        self.nb_time_points = nb_time_points_value
        self.time_array = np.linspace(0.0,
                                      self.total_time,
                                      self.nb_time_points)
        #this will contain the solution
        self.sol_c = [[] for i in range(self.nb_time_points)]
        #initialize first concentration profile
        self.sol_c[0] = np.array(self.c_init)
        # obtain the shape: r in polar coordinate,
        #and x and y in cartesian coordinates
        #nx, ny are the coordinates of the normal vector to the shape
        self.update_r_x_y_dl_nx_ny_metric_det_derivative()
        # Obtain profiles of kon and koff,
        self.update_kon_koff()
        #when initiating the simulation,
        #an option can be passed to change the shape
        self.change_shape = kwargs.get('change_shape', False)
        # a function can be passed which indicates how the shape is updated
        # get_new_shape is a function that takes the current time
        # and return information about the shape:
        # 1. ellipse_L
        # 2. ellipse_e
        # 3. d ellipse_L/dt
        # 4. d ellipse_e/dt
        self.get_new_shape = kwargs.get('get_new_shape', None)
    def update_centrosome_metaphase_plate(self):
        """ update positon of centrosomes and ends of metaphase plate"""
        [self.centrosome_position_1,
         self.centrosome_position_2,
         self.metaphase_plate_1,
         self.metaphase_plate_2] \
         = self.get_centrosome_metaphase_plate(self.spindle_angle)
    def get_centrosome_metaphase_plate(self, spindle_angle):
        """ function that gives the position of centrosome
        and ends of metaphase plate for a given value of the spindle angle
        useful to have this in general and for animations
        """
        centrosome_position_1 = np.array([(self.centrosome_nucleus_distance
                                           * np.cos(spindle_angle)),
                                          (self.centrosome_nucleus_distance
                                           * np.sin(spindle_angle))])
        centrosome_position_2 = np.array([(self.centrosome_nucleus_distance
                                           * np.cos(spindle_angle + np.pi)),
                                          (self.centrosome_nucleus_distance
                                           * np.sin(spindle_angle + np.pi))])
        # position of the ends of metaphase plate
        metaphase_plate_1 = np.array([(self.metaphase_plate_length / 2.0
                                       * np.cos(spindle_angle + np.pi / 2.0)),
                                      (self.metaphase_plate_length / 2.0
                                       * np.sin(spindle_angle + np.pi / 2.0))])
        metaphase_plate_2 = np.array([(self.metaphase_plate_length / 2.0
                                       * np.cos(spindle_angle - np.pi / 2.0)),
                                      (self.metaphase_plate_length / 2.0
                                       * np.sin(spindle_angle - np.pi / 2.0))])
        return [centrosome_position_1,
                centrosome_position_2,
                metaphase_plate_1,
                metaphase_plate_2]
# =============================================================================
# FUNCTIONS TO CALCULATE GEOMETRICAL QUANTITIES
# =============================================================================
    def r(self, theta, ellipse_L, ellipse_e):
        """return the radius of cell contour position in polar coordinates"""
        return ellipse_L / np.sqrt(1.0 - np.square(ellipse_e * np.cos(theta)))
    def distance_centrosome(self, theta, spindle_angle, ellipse_L, ellipse_e):
        """return the distance to one of the two centrosomes
        (centrosome located at angle phi)
        of a point located at angle theta
        on the cell contour
        """
        return np.sqrt(np.square(self.r(theta, ellipse_L, ellipse_e))
                       + np.square(self.centrosome_nucleus_distance)
                       - 2.0 * self.r(theta, ellipse_L, ellipse_e)
                       * self.centrosome_nucleus_distance
                       * np.cos(theta - spindle_angle))
    def get_r_x_y_nx_ny_dldtheta(self, ellipse_L, ellipse_e):
        """ calculate the shape and normal vector
        r is the radial value in polar coordinates
        x,y the positions in cartesian coordinates
        nx, ny the coordinates of the normal vector to the shape
        dldtheta is the length element (sqrt(r^2+r'^2))
        """
        r_array = np.array([self.r(theta, ellipse_L, ellipse_e)
                   for theta in self.theta_array])
        x_array = r_array * np.cos(self.theta_array)
        y_array = r_array * np.sin(self.theta_array)
        drdtheta_array = -(np.square(ellipse_e)
                           *ellipse_L
                           *np.sin(self.theta_array)
                           *np.cos(self.theta_array)
                           /np.power(1.0
                                     - np.square(ellipse_e
                                                 *np.cos(self.theta_array)),
                                     3.0/2.0)
                          )
        dldtheta_array = np.sqrt(np.square(r_array) + np.square(drdtheta_array))
        nx_array = 1.0 / dldtheta_array * (r_array
                                           * np.cos(self.theta_array)
                                           + drdtheta_array
                                           * np.sin(self.theta_array))
        ny_array = 1.0 / dldtheta_array * (r_array
                                           * np.sin(self.theta_array)
                                           - drdtheta_array
                                           * np.cos(self.theta_array))
        return [r_array, x_array, y_array, nx_array, ny_array, dldtheta_array]
    def get_r_x_y_dl_nx_ny_metric_det_derivative(self, ellipse_L, ellipse_e):
        """ calculate the shape position,
        length element, normal vector,
        derivative of the metric determinant
        r is the radial value in polar coordinates
        x,y the positions in cartesian coordinates
        dldtheta is the length element (sqrt(r^2+r'^2))
        nx, ny are the coordinates of the normal vector to the shape
        metric_det_derivative is the derivative of the metric determinant
        """
        [r_array,
         x_array,
         y_array,
         nx_array,
         ny_array,
         dldtheta_array] = self.get_r_x_y_nx_ny_dldtheta(ellipse_L, ellipse_e)
        #the derivative of the metric determinant is useful for the Laplace-Beltrami operator
        metric_det_derivative = - ((np.square(ellipse_e * ellipse_L)
                                    * np.sin(2.0 * self.theta_array)
                                    * (1.0 - np.square(ellipse_e)
                                       + np.power(ellipse_e, 4)
                                       + np.square(ellipse_e)
                                       * (np.square(ellipse_e) - 2.0)
                                       * np.cos(2.0 * self.theta_array)
                                      )
                                   )
                                   /np.power(1.0 - np.square(ellipse_e
                                                           *np.cos(self.theta_array)),
                                             4)
                                  )
        return [r_array,
                x_array,
                y_array,
                dldtheta_array,
                nx_array,
                ny_array,
                metric_det_derivative]
    def update_r_x_y_dl_nx_ny_metric_det_derivative(self):
        """update geometric quantities """
        [self.r_array,
         self.x_array,
         self.y_array,
         self.dldtheta_array,
         self.nx_array,
         self.ny_array,
         self.metric_det_derivative] = self.get_r_x_y_dl_nx_ny_metric_det_derivative(self.ellipse_L,
                                                                                     self.ellipse_e)
    def calculate_distance_metaphase_plate_array(self,
                                                 metaphase_plate_1,
                                                 metaphase_plate_2,
                                                 x_array,
                                                 y_array):
        """ return a vector of closest distance for each point of the contour
        to the metaphase plate,
        represented by a segment
        (or a circle if the condition circular_metaphase_plate is true)
        """
        if self.circular_metaphase_plate:
            #here the center of the metaphase plate is set to (0,0) and the radius
            #to metaphase_plate_length / 2.0
            distance_array = np.array([distance_circle_point([0.0, 0.0],
                                                             self.metaphase_plate_length/2.0,
                                                             np.array(list(P_vect)))
                                       for P_vect
                                       in zip(x_array,
                                              y_array)])
            return distance_array
        #P_vect is a point on the cell contour
        distance_array = np.array([distance_segment_point(metaphase_plate_1,
                                                          metaphase_plate_2,
                                                          np.array(list(P_vect)))
                                   for P_vect
                                   in zip(x_array,
                                          y_array)])
        return distance_array

    def calculate_distance_centrosomes(self,
                                       centrosome_position_1,
                                       centrosome_position_2,
                                       x_array,
                                       y_array):
        """return the distance to each centrosome"""
        distance_centrosome_1 = np.sqrt(np.square(x_array
                                                  - centrosome_position_1[0])
                                        +np.square(y_array
                                                   - centrosome_position_1[1]))
        distance_centrosome_2 = np.sqrt(np.square(x_array
                                                  - centrosome_position_2[0])
                                        +np.square(y_array
                                                   - centrosome_position_2[1]))
        return([distance_centrosome_1,
                distance_centrosome_2])
# =============================================================================
# FUNCTIONS TO CALCULATE THE SOLUTION
# =============================================================================
    def get_kon_koff(self,
                     metaphase_plate_1,
                     metaphase_plate_2,
                     x_array,
                     y_array):
        """ function that calculates the profile of
        kon, koff
        """
        #obtain a vector which for each point gives the closest distance to the metaphase plate
        distance_array = self.calculate_distance_metaphase_plate_array(metaphase_plate_1,
                                                                       metaphase_plate_2,
                                                                       x_array,
                                                                       y_array)
        k_on_array = (self.k_on_far
                      * np.array(distance_array > self.nucleus_inhibition_range)
                      + self.k_on_near
                      * np.array(distance_array <= self.nucleus_inhibition_range))
        k_off_array = (self.k_off_far
                       * np.array(distance_array > self.nucleus_inhibition_range)
                       + self.k_off_near
                       * np.array(distance_array <= self.nucleus_inhibition_range))
        return [k_on_array, k_off_array]
    def update_kon_koff(self):
        """ update kon, koff"""
        [self.k_on_array,
         self.k_off_array] = self.get_kon_koff(self.metaphase_plate_1,
                                               self.metaphase_plate_2,
                                               self.x_array,
                                               self.y_array)
    def get_torque(self,
                   c_array,
                   spindle_angle,
                   ellipse_L,
                   ellipse_e):
        """calculate total torque 
        in practice calculate torque acting on centrosome 1
        and multiply by 2, due to overall symmetry.
        calculate the integral only in regions where the MTs
        touch the cell contour
        the concentration field is interpolated with a spline
        """
        theta_array_extended = np.concatenate((self.theta_array,
                                               [self.theta_array[0]
                                                + 2.0 * np.pi]))
        c_array_extended = np.concatenate((c_array, [c_array[0]]))
        c_spline = CubicSpline(theta_array_extended,
                               c_array_extended,
                               bc_type='periodic')
        list_extremal_points = self.find_extremal_points_MTs(spindle_angle,
                                                             ellipse_L,
                                                             ellipse_e)
        #distance to the extremal MT of the point at theta=0
        #if positive, indicates that the pole at theta=0 is not touching any MT
        distance_MT_theta_0 = (self.distance_centrosome(0,
                                                        spindle_angle,
                                                        ellipse_L,
                                                        ellipse_e)
                              - self.length_MTs)
        integrand_function \
        = lambda this_theta: (c_spline(this_theta)
                              * self.get_torque_geometric_integrand(this_theta,
                                                                    spindle_angle,
                                                                    ellipse_L,
                                                                    ellipse_e))
        #distinguishes cases according to how many segments must be integrated
        if len(list_extremal_points) == 0:
            if distance_MT_theta_0 > 0:
                #case where the MTs are nowhere touching the cortex
                return 0
            #case where the MTs are touching the cortex everywhere
            torque, _ = sp.integrate.quad(integrand_function,
                                          0.0,
                                          2.0 * np.pi,
                                          points=theta_array_extended,#helps the integration
                                          limit=5000)
        if len(list_extremal_points) == 2:
            if distance_MT_theta_0 > 0:
                torque, _ = sp.integrate.quad(integrand_function,
                                              list_extremal_points[0],
                                              list_extremal_points[1],
                                              points=theta_array_extended,
                                              limit=5000)
            else:
                torque1, _ = sp.integrate.quad(integrand_function,
                                               0.0,
                                               list_extremal_points[0],
                                               points=theta_array_extended,
                                               limit=5000)
                torque2, _ = sp.integrate.quad(integrand_function,
                                               list_extremal_points[1],
                                               2.0 * np.pi,
                                               points=theta_array_extended,
                                               limit=5000)
                torque = torque1 + torque2
        if len(list_extremal_points) == 4:
            if distance_MT_theta_0 > 0:
                torque1, _ = sp.integrate.quad(integrand_function,
                                               list_extremal_points[0],
                                               list_extremal_points[1],
                                               points=theta_array_extended,
                                               limit=5000)
                torque2, _ = sp.integrate.quad(integrand_function,
                                               list_extremal_points[2],
                                               list_extremal_points[3],
                                               points=theta_array_extended,
                                               limit=5000)
                torque = torque1 + torque2
            else:
                torque1, _ = sp.integrate.quad(integrand_function,
                                               0,
                                               list_extremal_points[0],
                                               points=theta_array_extended,
                                               limit=5000)
                torque2, _ = sp.integrate.quad(integrand_function,
                                               list_extremal_points[1],
                                               list_extremal_points[2],
                                               points=theta_array_extended,
                                               limit=5000)
                torque3, _ = sp.integrate.quad(integrand_function,
                                               list_extremal_points[3],
                                               2.0 * np.pi,
                                               points=theta_array_extended,
                                               limit=5000)
                torque = torque1 + torque2 + torque3
        return torque
    def get_torque_geometric_integrand(self, theta, spindle_angle, ellipse_L, ellipse_e):
        """ return the geometric part of the integrand
        in the torque calculation for the angle theta in polar coordinates
        There is also a factor 2 arising
        from the summation of the two spindles
        Overall compared to the supplementary this calculates 2 * h(theta,phi)"""
        #local radius as given by ellipse
        r = self.r(theta, ellipse_L, ellipse_e)
        dr = - ((np.square(ellipse_e) *  ellipse_L * np.sin(theta) * np.cos(theta))
                / np.power(1.0 - np.square(ellipse_e * np.cos(theta)), 3.0/2.0))
        #to make the notation shorter
        lc = self.centrosome_nucleus_distance
        dc = np.sqrt(np.square(r)
                     + np.square(lc)
                     - 2.0 * r * lc * np.cos(theta - spindle_angle))
        torque_integrand = (2.0 * (- lc * np.cos(theta - spindle_angle) * r
                                        + np.square(r)
                                        - lc * np.sin(theta - spindle_angle) * dr)
                            / np.power(dc, 3) * r * np.sin(theta - spindle_angle))
        return torque_integrand
    def find_extremal_points_MTs(self, spindle_angle, ellipse_L, ellipse_e):
        """ return set of points where the astral MTs
        coming from centrosome 1
        stop/start touching the cell surface
        """
        #distance to centrosome 1, minus maximal MT length
        #if positive, point is not in contact with MT
        #if negative, point is in contact with MT
        distance_function = lambda theta: (self.distance_centrosome(theta,
                                                                    spindle_angle,
                                                                    ellipse_L,
                                                                    ellipse_e)
                                          - self.length_MTs)
        #first test if there is no contact or full contact
        min_value = minimize_scalar(distance_function,
                                    bounds=(0.0, 2.0 * np.pi),
                                    method='bounded').fun
        max_value = -1.0 * (minimize_scalar(lambda theta: - 1.0 * distance_function(theta),
                                           bounds=(0.0, 2.0 * np.pi),
                                           method='bounded').fun)
        #if the MTs never cross the cell surface, return an empty list
        if min_value > 0:
            return []
        #if the MTs are touching everywhere, return an empty list as well
        if max_value < 0:
            return []
        #now look for crossing points
        # first evaluate function at EXTREMAL_POINTS_SEARCH_RES
        # different points
        # this is a first rough search
        # crossing points that are close to each other
        # below this resolution (1/EXTREMAL_POINTS_SEARCH_RES)
        # will be missed
        U = np.linspace(0.0, 2.0 * np.pi, self.EXTREMAL_POINTS_SEARCH_RES)
        c = distance_function(U)
        s = np.sign(c)
        #list of zeros of the function = list of crossing points
        zeros = []
        for i in range(self.EXTREMAL_POINTS_SEARCH_RES - 1):
            #if one exactly hits zero  
            #the second condition excludes the case 
            #where there is a 0 without crossing
            #(not taken into account as a single contact point 
            #would lead to a zero force anyway)
            if s[i]==0 and s[i-1]+s[i+1]==0: 
                zeros.append(U[i])
            #if one exactly hits a zero at i=0 
            #because of periodicity the last item in s 
            #is the same as the first, so test for
            #crossing one needs to take s[i-2]
            if s[i]==0 and i==0 and s[i-2]+s[i+1]==0: 
                zeros.append(U[i])
            if s[i] + s[i+1] == 0: # opposite signs
                u = sp.optimize.brentq(distance_function, U[i], U[i+1])
                zeros.append(u)
        return zeros
    def dcdt_dphidt(self, cx, t):
        """Differential equations for the concentration equation,
        discretized in x,
        and for the evolution of the spindle angle equation.
        """
        #cx contains the concentration vector and the nucleus position,
        #so first extracts the concentration vector only
        c_array = cx[0:-1]
        #then update the angle of the spindle
        self.spindle_angle = cx[-1]
        if self.change_shape:
            [self.ellipse_L,
             self.ellipse_e,
             ellipse_dtL,
             ellipse_dte] = self.get_new_shape(t)
            #following this change of cell shape
            #recalculate shape-associated quantities
            self.update_r_x_y_dl_nx_ny_metric_det_derivative()
        # update correspondingly position of the centrosomes
        #and metaphase plate
        self.update_centrosome_metaphase_plate()
        # Calculate the profile of kon/koff
        self.update_kon_koff()
        #calculate the second derivative using spline interpolation
        theta_array_extended = np.concatenate((self.theta_array,
                                               [self.theta_array[0]
                                                +2.0 * np.pi]))
        c_array_extended = np.concatenate((c_array, [c_array[0]]))
        c_spline = CubicSpline(theta_array_extended,
                               c_array_extended,
                               bc_type='periodic')
        #compute the x derivatives with a spline
        #use the Laplace-Beltrami operator on the ellipse
        #cxx is the real Laplacian as defined by the Laplace-Beltrami operator
        cxx = (c_spline(self.theta_array, 2)
               / np.square(self.dldtheta_array)
               - c_spline(self.theta_array, 1)
               * self.metric_det_derivative
               / 2.0
               / np.power(self.dldtheta_array, 4)
              )
        # contribution of 1/sqrt(g) * d (sqrt(g))/dt
        #the quantities below X and Xovere are introduced for convenience
        Xovere = self.ellipse_e * np.square(np.cos(self.theta_array))
        X = np.square(self.ellipse_e * np.cos(self.theta_array))
        e2 = np.square(self.ellipse_e) #introduced for convenience
        factor = -(Xovere * (1.0 - 4.0 * X + e2 * (2.0 + X))
                   / (X - 1.0)
                   / (1.0 + (e2 - 2.0) * X))
        if self.change_shape:
            dtmetric = (ellipse_dtL / self.ellipse_L
                        + factor * ellipse_dte)
        else:
            dtmetric = 0
        #Compute dc/dt.
        dcdt_vector = (self.D * cxx
                       + self.k_on_array
                       - self.k_off_array * c_array
                       - dtmetric * c_array)
        #calculate overall torque acting on the spindle
        dphidt = (1.0 / self.tau_r
                  * self.get_torque(c_array,
                                    self.spindle_angle,
                                    self.ellipse_L,
                                    self.ellipse_e))
        return np.concatenate((dcdt_vector, np.array([dphidt])))
    def solve_equations(self):
        """Use odeint to solve the concentration equation on a periodic domain.
        and evolve the spindle angle
        Here tau_r is constant in time
        """
        #the second argument of odeint is the initial condition
        initial_condition = np.concatenate((self.c_init,
                                            np.array([self.spindle_angle])))
        sol = odeint(self.dcdt_dphidt,
                     initial_condition,
                     self.time_array,
                     mxstep=500000,
                     rtol=10e-10,
                     atol=10e-10,
                     hmax=0.01)
        self.sol_c = np.delete(sol, -1, axis=1)
        self.spindle_angle_vect = sol[:, -1]
        #also save the shape of the cell during the simulation
        if not self.change_shape:
            self.ellipse_L_vect = (self.ellipse_L
                                   * np.ones(self.spindle_angle_vect.shape))
            self.ellipse_e_vect = (self.ellipse_e
                                   * np.ones(self.spindle_angle_vect.shape))
            self.tau_r_vect = np.array([self.tau_r
                                        for this_time
                                        in self.time_array])
            self.circular_metaphase_plate_vect = np.array([self.circular_metaphase_plate
                                                           for this_time
                                                           in self.time_array])
        else:
            #perform a final update of the shape
            [self.ellipse_L,
             self.ellipse_e,
             _,
             _] = self.get_new_shape(self.time_array[-1])
            #following this change of cell shape,
            #recalculate shape-associated quantities
            self.update_r_x_y_dl_nx_ny_metric_det_derivative()
            self.ellipse_L_vect = np.array([self.get_new_shape(this_time)[0]
                                            for this_time in self.time_array])
            self.ellipse_e_vect = np.array([self.get_new_shape(this_time)[1]
                                            for this_time in self.time_array])
            self.tau_r_vect = np.array([self.tau_r
                                        for this_time in self.time_array])
            self.circular_metaphase_plate_vect = np.array([self.circular_metaphase_plate
                                                           for this_time
                                                           in self.time_array])
        return [self.sol_c, self.spindle_angle_vect]
    def solve_equations_two_steps(self, time_between_two_steps, tau_r):
        """Use odeint to solve the concentration equation on a periodic domain.
        and evolve the spindle angle
        do it in two steps: 1. in the first step for times smaller than
        time_between_two_steps, the metaphase
        plate is taken to be a circle and the spindle does not rotate
        2. in the second step for times larger than time_between_two_steps,
        the metaphase plate is a line
        and the spindle does rotate with characteristic time tau_r
        """
        time_array_1 = [this_time
                        for this_time
                        in self.time_array
                        if this_time <= time_between_two_steps]
        #for the second time array, the last time point is included
        #this is because one needs to reimplement the last step
        #as an initial condition before
        #continuing the iteration
        time_array_2 = np.concatenate(([time_array_1[-1]],
                                       [this_time
                                        for this_time
                                        in self.time_array
                                        if this_time > time_between_two_steps]))
        # FIRST STEP
        #the spindle can not rotate so tau_r is set to infinity
        self.tau_r = np.inf
        #here the metaphase plate is a circle
        self.circular_metaphase_plate = True
        #the second argument of odeint is the initial condition
        initial_condition = np.concatenate((self.c_init,
                                            np.array([self.spindle_angle])))

        sol = odeint(self.dcdt_dphidt,
                     initial_condition,
                     time_array_1,
                     mxstep=500000,
                     rtol=10e-10,
                     atol=10e-10)
        self.sol_c_1 = np.delete(sol, -1, axis=1)
        self.spindle_angle_vect_1 = sol[:, -1]
        # SECOND STEP
        #now the spindle can rotate
        self.tau_r = tau_r
        #the metaphase plate is now a line, default choice
        self.circular_metaphase_plate = False
        initial_condition = np.transpose(sol[-1, :])
        sol = odeint(self.dcdt_dphidt,
                     initial_condition,
                     time_array_2,
                     mxstep=500000,
                     rtol=10e-10,
                     atol=10e-10)
        #does not retain the first value as it was set at the last value of sol_c_1
        #hence the vectors are taken from index 1 instead of 0
        self.sol_c_2 = np.delete(sol[1:, :], -1, axis=1)
        self.spindle_angle_vect_2 = sol[1:, -1]
        # FULL SOLUTION OBTAINED BY CONCATENATION
        self.sol_c = np.concatenate((self.sol_c_1, self.sol_c_2))
        self.spindle_angle_vect = np.concatenate((self.spindle_angle_vect_1,
                                                  self.spindle_angle_vect_2))
        #also save the shape of the cell during the simulation
        if not self.change_shape:
            #this is not used in practice.
            self.ellipse_L_vect = (self.ellipse_L
                                   * np.ones(self.spindle_angle_vect.shape))
            self.ellipse_e_vect = (self.ellipse_e
                                   * np.ones(self.spindle_angle_vect.shape))
            self.tau_r_vect = np.array([tau_r
                                        for this_time
                                        in self.time_array])
            self.circular_metaphase_plate_vect = np.array([False
                                                           for this_time
                                                           in self.time_array])
        else:
            #perform a final update of the shape
            [self.ellipse_L,
             self.ellipse_e,
             _,
             _] = self.get_new_shape(self.time_array[-1])
            # recalculate shape-associated quantities that changed during the 
            # simulation
            self.update_r_x_y_dl_nx_ny_metric_det_derivative()
            self.ellipse_L_vect = np.array([self.get_new_shape(this_time)[0]
                                            for this_time in self.time_array])
            self.ellipse_e_vect = np.array([self.get_new_shape(this_time)[1]
                                            for this_time in self.time_array])
            self.tau_r_vect = np.array([tau_r
                                        if (this_time > time_between_two_steps)
                                        else np.inf
                                        for this_time in self.time_array])
            self.circular_metaphase_plate_vect = np.array([not (this_time > time_between_two_steps)
                                                           for this_time in self.time_array])
        return [self.sol_c, self.spindle_angle_vect]
    def calculate_geometry_from_solution(self):
        """once the solution is calculated,
        obtain a set of vectors containing the
        geometrical parameters
        useful to plot and animate the solution
        """
        self.centrosome_position_1_vect = []
        self.centrosome_position_2_vect = []
        self.metaphase_plate_1_vect = []
        self.metaphase_plate_2_vect = []
        self.k_off_array_vect = []
        self.torque_vect = []
        #for every angle
        #calculate the positions of centrosome and metaphase plate
        for (this_c_array,
             this_spindle_angle,
             this_ellipse_L,
             this_ellipse_e,
             this_circular_metaphase_plate) in zip(self.sol_c,
                                                   self.spindle_angle_vect,
                                                   self.ellipse_L_vect,
                                                   self.ellipse_e_vect,
                                                   self.circular_metaphase_plate_vect):
            [centrosome_position_1,
             centrosome_position_2,
             metaphase_plate_1,
             metaphase_plate_2] = self.get_centrosome_metaphase_plate(this_spindle_angle)
            self.centrosome_position_1_vect.append(centrosome_position_1)
            self.centrosome_position_2_vect.append(centrosome_position_2)
            self.metaphase_plate_1_vect.append(metaphase_plate_1)
            self.metaphase_plate_2_vect.append(metaphase_plate_2)
            [_,
             x_array,
             y_array,
             nx_array,
             ny_array,
             _] = self.get_r_x_y_nx_ny_dldtheta(this_ellipse_L, this_ellipse_e)
            self.circular_metaphase_plate = this_circular_metaphase_plate
            [_, k_off_array] = self.get_kon_koff(metaphase_plate_1,
                                                 metaphase_plate_2,
                                                 x_array,
                                                 y_array)
            self.k_off_array_vect.append(k_off_array)
            self.torque_vect.append(self.get_torque(this_c_array,
                                                    this_spindle_angle,
                                                    this_ellipse_L,
                                                    this_ellipse_e))
# =============================================================================
#     FUNCTIONS TO PLOT THE SOLUTION
# =============================================================================
    def plot_c_profile(self):
        """ plot subfigures with concentration profiles
        and spindle angle over time"""
        fig = plt.figure(1)
        plt.clf()
        ax = fig.subplots(nrows=2, ncols=2)
        plt.subplots_adjust(hspace=0.5, wspace=0.5)
        ax[0, 0].set_title('concentration profile at different times',
                           fontsize='medium')
        ax[0, 0].set_xlabel('angle', fontsize='small')
        ax[0, 0].set_ylabel('normalized concentration',
                            fontsize='small')
        for c_vect in self.sol_c:
            ax[0, 0].plot(self.theta_array, c_vect)
        ax[0, 1].set_title('spindle angle', fontsize='medium')
        ax[0, 1].set_xlabel('time (min)', fontsize='small')
        ax[0, 1].set_ylabel('angle', fontsize='small')
        ax[0, 1].scatter(self.time_array, self.spindle_angle_vect)
        spindle_angular_velocity = (np.diff(self.spindle_angle_vect)
                                    / np.diff(self.time_array))
        ax[1, 0].scatter(0.5*(self.time_array[0:-1]+self.time_array[1:]),
                         spindle_angular_velocity)
        ax[1, 0].scatter(self.time_array,
                         (1.0 / (self.tau_r_vect)
                          * np.array(self.torque_vect)), color='red')
        ax[1, 0].set_title('spindle angular velocity (blue) and torque (red)',
                           fontsize='medium')
        ax[1, 0].set_xlabel('time in minutes', fontsize='small')
        ax[1, 0].set_ylabel('velocity (min^(-1))', fontsize='small')
        plt.show()
    def plot_kymograph(self):
        """ plot kymograph"""
        fig = plt.figure(2)
        ax = fig.subplots()
        max_concentration = np.max(np.abs(self.sol_c))
        kymograph_plot = ax.imshow(np.transpose(self.sol_c),
                                   clim=[0, max_concentration],
                                   aspect='auto',
                                   extent=[0, self.total_time, 0, 2.0*np.pi])
        cbar = ax.figure.colorbar(kymograph_plot, ax=ax)
        cbar.ax.set_ylabel('concentration', rotation=-90, va="bottom")
        #ax.set_xticks(np.linspace(0,self.total_time,3))
        #ax.set_yticks(np.linspace(0,self.L,3))
        ax.set_xlabel('time (min)')
        ax.set_ylabel('x position (um)')
        ax.set_title('kymograph of concentration')
        plt.show()
    def plot_cell(self, this_time_index):
        """ plotting the cell shape
        and the spindle at time this_time_index
        """
        self.fig = plt.figure(3)
        plt.clf()
        #define subplots
        ax = self.fig.subplots(nrows=2, ncols=2)
        plt.subplots_adjust(hspace=0.5, wspace=1.0)
        #some information about the current shape
        ellipse_L = self.ellipse_L_vect[this_time_index]
        ellipse_e = self.ellipse_e_vect[this_time_index]
        max_ellipse_L = max(self.ellipse_L_vect)
        #define a number of objects that
        #will be plotted in different combinations
        #draw cell shape
        ellipse_length = ellipse_L / np.sqrt(1-np.square(ellipse_e))
        ellipse_height = ellipse_L
        spindle_angle = self.spindle_angle_vect[this_time_index]
        [_, x_array, y_array, _, _,_] = self.get_r_x_y_nx_ny_dldtheta(ellipse_L,
                                                                      ellipse_e)
        circular_metaphase_plate = self.circular_metaphase_plate_vect[this_time_index]
        #when a solution has been calculated,
        #returns the centrosome and metaphase plate position at time 0
        centrosome_position_1 = self.centrosome_position_1_vect[this_time_index]
        centrosome_position_2 = self.centrosome_position_2_vect[this_time_index]
        metaphase_plate_1 = self.metaphase_plate_1_vect[this_time_index]
        metaphase_plate_2 = self.metaphase_plate_2_vect[this_time_index]
        x_range = [-2.0 * max_ellipse_L, 2.0 * max_ellipse_L]
        y_range = [-2.0 * max_ellipse_L, 2.0 * max_ellipse_L]
        for i in range(2):
            for j in range(2):
                #draw spindle as a polygon joining the metaphase plate
                #and the centrosome
                Spindle_array = np.array([centrosome_position_1,
                                          metaphase_plate_1,
                                          centrosome_position_2,
                                          metaphase_plate_2])
                cell_shape = Ellipse(xy=(0, 0),
                                     width=2.0*ellipse_length,
                                     height=2.0*ellipse_height,
                                     angle=0,
                                     edgecolor='None',
                                     fc='whitesmoke',
                                     lw=2)
                Spindle_patch = plt.Polygon(Spindle_array,
                                            color='lightgreen',
                                            lw=3)
                #draw microtubules
                MT_1_patch = Ellipse(xy=(centrosome_position_1[0],
                                         centrosome_position_1[1]),
                                     width=2.0 * self.length_MTs,
                                     height=2.0 * self.length_MTs,
                                     edgecolor='None',
                                     fc='lightgreen',
                                     alpha=0.4)
                MT_2_patch = Ellipse(xy=(centrosome_position_2[0],
                                         centrosome_position_2[1]),
                                     width=2.0 * self.length_MTs,
                                     height=2.0 * self.length_MTs,
                                     edgecolor='None',
                                     fc='lightgreen',
                                     alpha=0.4)
                #actually do the drawing
                ax[i, j].add_patch(cell_shape)
                #ensures that MTs are only drawn inside the cell
                MT_2_patch.set_clip_path(cell_shape)
                ax[i, j].add_patch(MT_2_patch)
                ax[i, j].add_patch(Spindle_patch)
                #ensures that MTs are only drawn inside the cell
                MT_1_patch.set_clip_path(cell_shape)
                ax[i, j].add_patch(MT_1_patch)
                ax[i, j].set_xlim(x_range)
                ax[i, j].set_ylim(y_range)
                ax[i, j].set_aspect(1)
        # FIRST PLOT add a cell contour to the first plot
        cell_shape_contour = Ellipse(xy=(0, 0),
                                     width=2.0 * ellipse_length,
                                     height=2.0 * ellipse_height,
                                     angle=0,
                                     edgecolor='black',
                                     fc='None',
                                     lw=2)
        ax[0, 0].add_patch(cell_shape_contour)
        # SECOND PLOT add a color code on the contour to the second plot
        #corresponding to LGN concentration
        points = np.array([x_array, y_array]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points, np.roll(points, -1, axis=0)], axis=1)
        #this vector contains the color code to plot
        #- for now the initial concentration
        color_to_plot = np.array(self.sol_c[this_time_index])
        # Create a continuous norm to map from data points to colors
        norm = plt.Normalize(0, 4)
        lc = LineCollection(segments, cmap='Reds', norm=norm)
        # Set the values used for colormapping
        lc.set_array(color_to_plot)
        lc.set_linewidth(2)
        line = ax[0, 1].add_collection(lc)
        self.fig.colorbar(line, ax=ax[0, 1])
        #self.fig.colorbar(line, ax=ax[0, 1])
        #ax[0, 1].set_title('LGN concentration')
        # THIRD AND FOURTH PLOT plot the region of RAN inhibition
        for k in range(2):
            cell_shape_contour = Ellipse(xy=(0, 0),
                                         width=2.0 * ellipse_length,
                                         height=2.0 * ellipse_height,
                                         angle=0,
                                         edgecolor='black',
                                         fc='None',
                                         lw=2)
            ax[1, k].add_patch(cell_shape_contour)
            if not circular_metaphase_plate:
                Ran_inhibition_patch_1 = Ellipse(xy=(metaphase_plate_1[0],
                                                     metaphase_plate_1[1]),
                                                 width=2.0 * self.nucleus_inhibition_range,
                                                 height=2.0 * self.nucleus_inhibition_range,
                                                 angle=0,
                                                 edgecolor='None',
                                                 fc='lightblue',
                                                 alpha=1.0)
                Ran_inhibition_patch_2 = Ellipse(xy=(metaphase_plate_2[0],
                                                     metaphase_plate_2[1]),
                                                 width=2.0 * self.nucleus_inhibition_range,
                                                 height=2.0 * self.nucleus_inhibition_range,
                                                 angle=0,
                                                 edgecolor='None',
                                                 fc='lightblue',
                                                 alpha=1.0)
                Ran_inhibition_patch_3 = Rectangle(xy=((self.nucleus_inhibition_range
                                                        *np.cos(spindle_angle)
                                                        +self.metaphase_plate_length/2.0
                                                        *np.sin(spindle_angle)),
                                                       (self.nucleus_inhibition_range
                                                        *np.sin(spindle_angle)
                                                        -self.metaphase_plate_length/2.0
                                                        *np.cos(spindle_angle))),
                                                   height=2 * self.nucleus_inhibition_range,
                                                   width=self.metaphase_plate_length,
                                                   angle=180.0*(spindle_angle+np.pi/2.0)/np.pi,
                                                   edgecolor='None',
                                                   fc='lightblue',
                                                   alpha=1.0)
                Ran_inhibition_patch_1.set_clip_path(cell_shape_contour)
                Ran_inhibition_patch_2.set_clip_path(cell_shape_contour)
                Ran_inhibition_patch_3.set_clip_path(cell_shape_contour)
                ax[1, k].add_patch(Ran_inhibition_patch_1)
                ax[1, k].add_patch(Ran_inhibition_patch_2)
                ax[1, k].add_patch(Ran_inhibition_patch_3)
            else:
                Ran_inhibition_patch_circular = Ellipse(xy=(0, 0),
                                                        width=2.0 * (self.metaphase_plate_length / 2.0
                                                                     + self.nucleus_inhibition_range),
                                                        height=2.0 * (self.metaphase_plate_length /2.0
                                                                      + self.nucleus_inhibition_range),
                                                        angle=0,
                                                        edgecolor='None',
                                                        fc='lightblue',
                                                        alpha=1.0)
                Ran_inhibition_patch_circular.set_clip_path(cell_shape_contour)
                ax[1, k].add_patch(Ran_inhibition_patch_circular)
        # FOURTH PLOT for verification, plot the profile of koff on the contour
        color_to_plot = self.k_off_array_vect[this_time_index]
        norm = plt.Normalize(self.k_off_far, self.k_off_near)
        lc = LineCollection(segments, cmap='plasma', norm=norm)
        # Set the values used for colormapping
        lc.set_array(color_to_plot)
        lc.set_linewidth(2)
        ax[1, 1].add_collection(lc)
        ax[1, 1].set_title('LGN koff')
        # METAPHASE PLATE
        #add the end (so that it appears on top)
        for i in range(2):
            for j in range(2):
                #draw metaphase plate
                if not circular_metaphase_plate:
                    metaphase_plate_line = mlines.Line2D([metaphase_plate_1[0],
                                                          metaphase_plate_2[0]],
                                                         [metaphase_plate_1[1],
                                                          metaphase_plate_2[1]],
                                                         color='black',
                                                         lw=3,
                                                         solid_capstyle='round')

                else:
                    metaphase_plate_line = Ellipse(xy=(0,
                                                       0),
                                                   width=self.metaphase_plate_length,
                                                   height=self.metaphase_plate_length,
                                                   color='black',
                                                   lw=3,
                                                   alpha=1.0)
                if not circular_metaphase_plate:
                    ax[i, j].add_line(metaphase_plate_line)
                else:
                    ax[i, j].add_patch(metaphase_plate_line)
        plt.show()
    def plot_cell_sequence(self, time_sequence):
        """ plotting the cell shape
        and the spindle at a sequence of time 
        contained in time_sequence
        """
        self.fig = plt.figure(5)
        plt.clf()
        #define subplots
        ax = self.fig.subplots(nrows=1, ncols=len(time_sequence))
        plt.subplots_adjust(hspace=0.1, wspace=0.)
        for i, this_time_index in enumerate(time_sequence):
            #some information about the current shape
            ellipse_L = self.ellipse_L_vect[this_time_index]
            ellipse_e = self.ellipse_e_vect[this_time_index]
            max_ellipse_L = max(self.ellipse_L_vect)
            #define a number of objects that will be plotted in different combinations
            #draw cell shape
            ellipse_length = ellipse_L / np.sqrt(1.0 - np.square(ellipse_e))
            ellipse_height = ellipse_L
            [_, x_array, y_array, _, _,_] = self.get_r_x_y_nx_ny_dldtheta(ellipse_L, ellipse_e)
            circular_metaphase_plate = self.circular_metaphase_plate_vect[this_time_index]
            #when a solution has been calculated,
            #returns the centrosome and metaphase plate position at time 0
            centrosome_position_1 = self.centrosome_position_1_vect[this_time_index]
            centrosome_position_2 = self.centrosome_position_2_vect[this_time_index]
            metaphase_plate_1 = self.metaphase_plate_1_vect[this_time_index]
            metaphase_plate_2 = self.metaphase_plate_2_vect[this_time_index]
            x_range = [-2.0 * max_ellipse_L, 2.0 * max_ellipse_L]
            y_range = [-2.0 * max_ellipse_L, 2.0 * max_ellipse_L]
            #draw spindle as a polygon joining the metaphase plate and the centrosome
            Spindle_array = np.array([centrosome_position_1,
                                      metaphase_plate_1,
                                      centrosome_position_2,
                                      metaphase_plate_2])
            cell_shape = Ellipse(xy=(0, 0),
                                 width=2.0 * ellipse_length,
                                 height=2.0 * ellipse_height,
                                 angle=0,
                                 edgecolor='None',
                                 fc='whitesmoke',
                                 lw=2)
            Spindle_patch = plt.Polygon(Spindle_array,
                                        color='lightgreen',
                                        lw=3)
            #actually do the drawing
            ax[i].add_patch(cell_shape)
            ax[i].add_patch(Spindle_patch)
            #uncomment to draw astral microtubules
#            MT_1_patch = Ellipse(xy=(centrosome_position_1[0],
#                                     centrosome_position_1[1]),
#                                 width=2.0 * self.length_MTs,
#                                 height=2.0 * self.length_MTs,
#                                 edgecolor='None',
#                                 fc='lightgreen',
#                                 alpha=0.4)
#            MT_2_patch = Ellipse(xy=(centrosome_position_2[0],
#                                     centrosome_position_2[1]),
#                                 width=2.0 * self.length_MTs,
#                                 height=2.0 * self.length_MTs,
#                                 edgecolor='None',
#                                 fc='lightgreen',
#                                 alpha=0.4)
#
#            ax[i].add_patch(MT_1_patch)
#            #ensures that MTs are only drawn inside the cell
#            MT_2_patch.set_clip_path(cell_shape)
#            ax[i].add_patch(MT_2_patch)  
#            #ensures that MTs are only drawn inside the cell
#            MT_1_patch.set_clip_path(cell_shape)
            ax[i].set_xlim(x_range)
            ax[i].set_ylim(y_range)
            ax[i].set_aspect(1)
            # LGN CONCENTRATION on the contour
            # with a colour code
            points = np.array([x_array, y_array]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points,
                                       np.roll(points, -1, axis=0)],
                                      axis=1)
            #this vector contains the color code to plot
            color_to_plot = np.array(self.sol_c[this_time_index])
            # Create a continuous norm to map from data points to colors
            norm = plt.Normalize(0, 4)
            lc = LineCollection(segments, cmap='Reds', norm=norm)
            # Set the values used for colormapping
            lc.set_array(color_to_plot)
            lc.set_linewidth(2)
            line = ax[i].add_collection(lc)
            # METAPHASE PLATE
            # add at the end (so that it appears on top)
            if not circular_metaphase_plate:
                metaphase_plate_line = mlines.Line2D([metaphase_plate_1[0],
                                                      metaphase_plate_2[0]],
                                                     [metaphase_plate_1[1],
                                                      metaphase_plate_2[1]],
                                                     color='black',
                                                     lw=3,
                                                     solid_capstyle='round')
            else:
                metaphase_plate_line = Ellipse(xy=(0,
                                                   0),
                                               width=self.metaphase_plate_length,
                                               height=self.metaphase_plate_length,
                                               color='black',
                                               lw=3,
                                               alpha=1.0)
            if not circular_metaphase_plate:
                ax[i].add_line(metaphase_plate_line)
            else:
                ax[i].add_patch(metaphase_plate_line)
            ax[i].axis('off')
        self.fig.subplots_adjust(right=0.9)
        cbar_ax = self.fig.add_axes([0.9, 0.45, 0.01, 0.1])
        #add a color bar to the figure
        self.fig.colorbar(line, cax=cbar_ax, aspect=10, ticks=[0, 4])
        plt.show()
    def animate(self):
        """trigger an animation to show
        the evolution of the concentration profile"""
        self.fig = plt.figure(4)
        plt.clf()
        x_range = [self.theta_array[0],
                   self.theta_array[-1]]
        y_range = [np.min(self.sol_c),
                   1.5*np.max(self.sol_c)]
        self.ax = plt.axes()
        self.ax.set_xlim(x_range)
        self.ax.set_ylim(y_range)
        plt.title('concentration field profile')
        plt.xlabel('angle theta')
        plt.ylabel('concentration')
        self.scat_conc_profile = plt.scatter(self.theta_array,
                                             self.sol_c[0],
                                             color='b',
                                             animated=True)
        plt.show()
        self.ani = animation.FuncAnimation(self.fig,
                                           func=self.update,
                                           frames=np.arange(1, len(self.time_array)),
                                           init_func=None,
                                           repeat=False,
                                           interval=10,
                                           blit=True)
        plt.show()
    def update(self, i):
        """ update function for animation of concentration profile"""
        self.scat_conc_profile.set_offsets(np.transpose(np.stack([self.theta_array,
                                                                  self.sol_c[i]])))
        return self.scat_conc_profile,
    def animate_cell(self):
        """trigger an animation showing the entire cell
        This function is very long because
        4 subplots are animated simultaneously
        to show separately the different elemnts
        and each element of each subplot
        must be animated separately
        """
        self.fig = plt.figure(3)
        plt.clf()
        #define subplots
        self.ax = self.fig.subplots(nrows=2, ncols=2)
        plt.subplots_adjust(hspace=0.5, wspace=1.0)
        #define a number of objects that will be plotted in different combinations
        #for all evaluations, take the first value in the list
        #as this is the initial time
        ellipse_L = self.ellipse_L_vect[0]
        ellipse_e = self.ellipse_e_vect[0]
        ellipse_length = ellipse_L / np.sqrt(1-np.square(ellipse_e))
        ellipse_height = ellipse_L
        spindle_angle = self.spindle_angle_vect[0]
        [_, x_array, y_array, _, _, _] = self.get_r_x_y_nx_ny_dldtheta(ellipse_L, ellipse_e)
        #when a solution has been calculated,
        #returns the centrosome and metaphase plate position at time 0
        centrosome_position_1 = self.centrosome_position_1_vect[0]
        centrosome_position_2 = self.centrosome_position_2_vect[0]
        metaphase_plate_1 = self.metaphase_plate_1_vect[0]
        metaphase_plate_2 = self.metaphase_plate_2_vect[0]
        x_range = [-1.2 * ellipse_length, 1.2 * ellipse_length]
        y_range = [-1.2 * ellipse_length, 1.2 * ellipse_length]
        for i in range(2):
            for j in range(2):
                self.ax[i, j].set_xlim(x_range)
                self.ax[i, j].set_ylim(y_range)
                self.ax[i, j].set_aspect(1)
        Spindle_array = np.array([centrosome_position_1,
                                  metaphase_plate_1,
                                  centrosome_position_2,
                                  metaphase_plate_2])
        #### FIRST PLOT ####
        ## draw the spindle ##
        self.animation_patches_Spindle_patch_0_0 = plt.Polygon(Spindle_array,
                                                               color='lightgreen',
                                                               lw=3,
                                                               animated=True)
        self.ax[0, 0].add_patch(self.animation_patches_Spindle_patch_0_0)
        ## draw the MTs ##
        self.animation_patches_MT_1_patch_0_0 = Ellipse(xy=(centrosome_position_1[0],
                                                            centrosome_position_1[1]),
                                                        width=2.0 * self.length_MTs,
                                                        height=2.0 * self.length_MTs,
                                                        edgecolor='None',
                                                        fc='lightgreen',
                                                        alpha=0.4,
                                                        animated=True)
        #to only see the spindle for clarity
        self.animation_patches_MT_1_patch_0_0.set_visible(False)
        self.ax[0, 0].add_patch(self.animation_patches_MT_1_patch_0_0)
        self.animation_patches_MT_2_patch_0_0 = Ellipse(xy=(centrosome_position_2[0],
                                                            centrosome_position_2[1]),
                                                        width=2.0 * self.length_MTs,
                                                        height=2.0 * self.length_MTs,
                                                        edgecolor='None',
                                                        fc='lightgreen',
                                                        alpha=0.4,
                                                        animated=True)
        #to only see the spindle for clarity
        self.animation_patches_MT_2_patch_0_0.set_visible(False)
        self.ax[0, 0].add_patch(self.animation_patches_MT_2_patch_0_0)
        ## draw the metaphase plate ##
        self.animation_patches_metaphase_plate_line_0_0 = mlines.Line2D([metaphase_plate_1[0],
                                                                         metaphase_plate_2[0]],
                                                                        [metaphase_plate_1[1],
                                                                         metaphase_plate_2[1]],
                                                                        color='black',
                                                                        lw=3,
                                                                        solid_capstyle='round',
                                                                        animated=True)
        self.ax[0, 0].add_line(self.animation_patches_metaphase_plate_line_0_0)
        self.animation_patches_metaphase_plate_circle_0_0 = Ellipse(xy=(0, 0),
                                                                    width=self.metaphase_plate_length,
                                                                    height=self.metaphase_plate_length,
                                                                    color='black',
                                                                    lw=3,
                                                                    alpha=1.0,
                                                                    animated=True)
        self.ax[0, 0].add_patch(self.animation_patches_metaphase_plate_circle_0_0)
        ## draw the cell shape contour ##
        self.animation_patches_cell_shape_contour_0_0 = Ellipse(xy=(0, 0),
                                                                width=2.0 * ellipse_length,
                                                                height=2.0 * ellipse_height,
                                                                angle=0,
                                                                edgecolor='black',
                                                                fc='None',
                                                                lw=2,
                                                                animated=True)
        self.ax[0, 0].add_patch(self.animation_patches_cell_shape_contour_0_0)
        ##### SECOND PLOT ####
        #add a color code on the contour to the second plot
        #corresponding to LGN concentration
        self.animation_cell_shape_0_1 = Ellipse(xy=(0, 0),
                                                width=2.0*ellipse_length,
                                                height=2.0*ellipse_height,
                                                angle=0,
                                                edgecolor='None',
                                                fc='whitesmoke',
                                                lw=2,
                                                animated=True)
        self.ax[0, 1].add_patch(self.animation_cell_shape_0_1)
        ## draw the spindle ##
        self.animation_patches_Spindle_patch_0_1 = plt.Polygon(Spindle_array,
                                                               color='lightgreen',
                                                               lw=3,
                                                               animated=True)
        self.ax[0, 1].add_patch(self.animation_patches_Spindle_patch_0_1)
        ## draw MTs ##
#        self.animation_patches_MT_1_patch_0_1 = Ellipse(xy=(centrosome_position_1[0],
#                                                            centrosome_position_1[1]),
#                                                        width=2.0 * self.length_MTs,
#                                                        height=2.0 * self.length_MTs,
#                                                        edgecolor='None',
#                                                        fc='lightgreen',
#                                                        alpha=0.4,
#                                                        animated=True)
#        self.ax[0, 1].add_patch(self.animation_patches_MT_1_patch_0_1)
#        self.animation_patches_MT_2_patch_0_1 = Ellipse(xy=(centrosome_position_2[0],
#                                                            centrosome_position_2[1]),
#                                                        width=2.0 * self.length_MTs,
#                                                        height=2.0 * self.length_MTs,
#                                                        edgecolor='None',
#                                                        fc='lightgreen',
#                                                        alpha=0.4,
#                                                        animated=True)
#        self.ax[0, 1].add_patch(self.animation_patches_MT_2_patch_0_1)
         ## draw metaphase plate##
        self.animation_patches_metaphase_plate_line_0_1 = mlines.Line2D([metaphase_plate_1[0],
                                                                         metaphase_plate_2[0]],
                                                                        [metaphase_plate_1[1],
                                                                         metaphase_plate_2[1]],
                                                                        color='black',
                                                                        lw=3,
                                                                        solid_capstyle='round',
                                                                        animated=True)
        self.animation_patches_metaphase_plate_circle_0_1 = Ellipse(xy=(0, 0),
                                                                    width=self.metaphase_plate_length,
                                                                    height=self.metaphase_plate_length,
                                                                    color='black',
                                                                    lw=3,
                                                                    alpha=1.0,
                                                                    animated=True)
        self.ax[0, 1].add_line(self.animation_patches_metaphase_plate_line_0_1)
        self.ax[0, 1].add_patch(self.animation_patches_metaphase_plate_circle_0_1)
        ## draw concentration profile ##
        points = np.array([x_array, y_array]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points, np.roll(points, -1, axis=0)], axis=1)
        #this vector contains the color code to plot
        #- for now the initial concentration
        color_to_plot = np.array(self.sol_c[0])
        # Create a continuous norm to map from data points to colors
        norm = plt.Normalize(0, 4)
        self.animation_patches_lc = LineCollection(segments,
                                                   cmap='Reds',
                                                   norm=norm,
                                                   animated=True)
        # Set the values used for colormapping
        self.animation_patches_lc.set_array(color_to_plot)
        self.animation_patches_lc.set_linewidth(2)
        line = self.ax[0, 1].add_collection(self.animation_patches_lc)
        self.fig.colorbar(line, ax=self.ax[0, 1])
        #self.ax[0, 1].set_title('LGN concentration')
#        #### THIRD PLOT ####
        #plot the region of RAN inhibition
        ## draw cell interior ##
        self.animation_cell_shape_1_0 = Ellipse(xy=(0, 0),
                                                width=2.0 * ellipse_length,
                                                height=2.0 * ellipse_height,
                                                angle=0,
                                                edgecolor='None',
                                                fc='whitesmoke',
                                                lw=2,
                                                animated=True)
        self.ax[1, 0].add_patch(self.animation_cell_shape_1_0)
        ## draw spindle ##
        self.animation_patches_Spindle_patch_1_0 = plt.Polygon(Spindle_array,
                                                               color='lightgreen',
                                                               lw=3,
                                                               animated=True)
        self.ax[1, 0].add_patch(self.animation_patches_Spindle_patch_1_0)
        ## draw MTs ##
        self.animation_patches_MT_1_patch_1_0 = Ellipse(xy=(centrosome_position_1[0],
                                                            centrosome_position_1[1]),
                                                        width=2.0 * self.length_MTs,
                                                        height=2.0 * self.length_MTs,
                                                        edgecolor='None',
                                                        fc='lightgreen',
                                                        alpha=0.4,
                                                        animated=True)
        self.ax[1, 0].add_patch(self.animation_patches_MT_1_patch_1_0)
        self.animation_patches_MT_2_patch_1_0 = Ellipse(xy=(centrosome_position_2[0],
                                                            centrosome_position_2[1]),
                                                        width=2.0 * self.length_MTs,
                                                        height=2.0 * self.length_MTs,
                                                        edgecolor='None',
                                                        fc='lightgreen',
                                                        alpha=0.4,
                                                        animated=True)
        self.ax[1, 0].add_patch(self.animation_patches_MT_2_patch_1_0)
        ## draw region of RAN inhibition
        self.animation_patches_Ran_inhibition_patch_1_1_0 \
        = Ellipse(xy=(metaphase_plate_1[0],
                      metaphase_plate_1[1]),
                  width=2.0 * self.nucleus_inhibition_range,
                  height=2.0 * self.nucleus_inhibition_range,
                  angle=0,
                  edgecolor='None',
                  fc='lightblue',
                  alpha=1.0,
                  animated=True)
        self.animation_patches_Ran_inhibition_patch_2_1_0 \
        = Ellipse(xy=(metaphase_plate_2[0],
                      metaphase_plate_2[1]),
                  width=2.0 * self.nucleus_inhibition_range,
                  height=2.0 * self.nucleus_inhibition_range,
                  angle=0,
                  edgecolor='None',
                  fc='lightblue',
                  alpha=1.0,
                  animated=True)
        self.animation_patches_Ran_inhibition_patch_3_1_0 \
        = Rectangle(xy=(((self.nucleus_inhibition_range
                           * np.cos(spindle_angle)
                           + self.metaphase_plate_length / 2.0
                           * np.sin(spindle_angle))),
                        ((self.nucleus_inhibition_range
                           * np.sin(spindle_angle)
                           - self.metaphase_plate_length / 2.0
                           * np.cos(spindle_angle)))),
                    height=2 * self.nucleus_inhibition_range,
                    width=self.metaphase_plate_length,
                    angle=180.0 * (spindle_angle + np.pi / 2.0) / np.pi,
                    edgecolor='None',
                    fc='lightblue',
                    alpha=1.0,
                    animated=True)
        self.animation_patches_Ran_inhibition_patch_circular_1_0 \
        = Ellipse(xy=(0, 0),
                  width=(2.0
                         * (self.metaphase_plate_length
                            / 2.0
                            + self.nucleus_inhibition_range)),
                  height=(2.0
                          * (self.metaphase_plate_length
                             / 2.0
                             + self.nucleus_inhibition_range)),
                  angle=0,
                  edgecolor='None',
                  fc='lightblue',
                  alpha=1.0,
                  animated=True)
        self.animation_patches_cell_shape_contour_1_0 \
        = Ellipse(xy=(0, 0),
                  width=2.0 * ellipse_length,
                  height=2.0 * ellipse_height,
                  angle=0,
                  edgecolor='black',
                  fc='None',
                  lw=2,
                  animated=True)
        self.ax[1, 0].add_patch(self.animation_patches_cell_shape_contour_1_0)
        self.animation_patches_Ran_inhibition_patch_1_1_0.set_clip_path(self.animation_patches_cell_shape_contour_1_0)
        self.animation_patches_Ran_inhibition_patch_2_1_0.set_clip_path(self.animation_patches_cell_shape_contour_1_0)
        self.animation_patches_Ran_inhibition_patch_3_1_0.set_clip_path(self.animation_patches_cell_shape_contour_1_0)
        self.ax[1, 0].add_patch(self.animation_patches_Ran_inhibition_patch_1_1_0)
        self.ax[1, 0].add_patch(self.animation_patches_Ran_inhibition_patch_2_1_0)
        self.ax[1, 0].add_patch(self.animation_patches_Ran_inhibition_patch_3_1_0)
        self.ax[1, 0].add_patch(self.animation_patches_Ran_inhibition_patch_circular_1_0)
        ## draw metaphase plate ##
        self.animation_patches_metaphase_plate_line_1_0 = mlines.Line2D([metaphase_plate_1[0],
                                                                         metaphase_plate_2[0]],
                                                                        [metaphase_plate_1[1],
                                                                         metaphase_plate_2[1]],
                                                                        color='black',
                                                                        lw=3,
                                                                        solid_capstyle='round',
                                                                        animated=True)
        self.animation_patches_metaphase_plate_circle_1_0 = Ellipse(xy=(0, 0),
                                                                    width=self.metaphase_plate_length,
                                                                    height=self.metaphase_plate_length,
                                                                    color='black',
                                                                    lw=3,
                                                                    alpha=1.0,
                                                                    animated=True)
        self.ax[1, 0].add_line(self.animation_patches_metaphase_plate_line_1_0)
        self.ax[1, 0].add_patch(self.animation_patches_metaphase_plate_circle_1_0)
        #### FOURTH PLOT ####
        #for verification, plot the profile of koff on the contour
        #draw Ran inhibition region
        self.animation_patches_Ran_inhibition_patch_1_1_1 \
        = Ellipse(xy=(metaphase_plate_1[0],
                      metaphase_plate_1[1]),
                  width=2.0 * self.nucleus_inhibition_range,
                  height=2.0 * self.nucleus_inhibition_range,
                  angle=0,
                  edgecolor='None',
                  fc='lightblue',
                  alpha=1.0,
                  animated=True)
        self.animation_patches_Ran_inhibition_patch_2_1_1 \
        = Ellipse(xy=(metaphase_plate_2[0],
                      metaphase_plate_2[1]),
                  width=2.0 * self.nucleus_inhibition_range,
                  height=2.0 * self.nucleus_inhibition_range,
                  angle=0,
                  edgecolor='None',
                  fc='lightblue',
                  alpha=1.0,
                  animated=True)
        self.animation_patches_Ran_inhibition_patch_3_1_1 \
        = Rectangle(xy=((self.nucleus_inhibition_range
                         * np.cos(self.spindle_angle)
                         + self.metaphase_plate_length
                         / 2.0
                         * np.sin(self.spindle_angle)),
                        (self.nucleus_inhibition_range
                         * np.sin(self.spindle_angle)
                         -self.metaphase_plate_length
                         / 2.0
                         * np.cos(self.spindle_angle))),
                    height=2 * self.nucleus_inhibition_range,
                    width=self.metaphase_plate_length,
                    angle=180.0 * (self.spindle_angle+np.pi/2.0)/np.pi,
                    edgecolor='None',
                    fc='lightblue',
                    alpha=1.0,
                    animated=True)
        self.animation_patches_Ran_inhibition_patch_circular_1_1 \
        = Ellipse(xy=(0, 0),
                  width=(2.0
                         * (self.metaphase_plate_length
                            / 2.0
                            + self.nucleus_inhibition_range)),
                  height=(2.0
                          * (self.metaphase_plate_length
                             / 2.0
                             + self.nucleus_inhibition_range)),
                  angle=0,
                  edgecolor='None',
                  fc='lightblue',
                  alpha=1.0,
                  animated=True)
        self.animation_patches_cell_shape_contour_1_1 \
        = Ellipse(xy=(0, 0),
                  width=2.0 * ellipse_length,
                  height=2.0 * ellipse_height,
                  angle=0,
                  edgecolor='black',
                  fc='None',
                  lw=2,
                  animated=True)
        self.animation_patches_Ran_inhibition_patch_1_1_1.set_clip_path(self.animation_patches_cell_shape_contour_1_1)
        self.animation_patches_Ran_inhibition_patch_2_1_1.set_clip_path(self.animation_patches_cell_shape_contour_1_1)
        self.animation_patches_Ran_inhibition_patch_3_1_1.set_clip_path(self.animation_patches_cell_shape_contour_1_1)
        self.ax[1, 1].add_patch(self.animation_patches_Ran_inhibition_patch_1_1_1)
        self.ax[1, 1].add_patch(self.animation_patches_Ran_inhibition_patch_2_1_1)
        self.ax[1, 1].add_patch(self.animation_patches_Ran_inhibition_patch_3_1_1)
        self.ax[1, 1].add_patch(self.animation_patches_Ran_inhibition_patch_circular_1_1)
        self.ax[1, 1].add_patch(self.animation_patches_cell_shape_contour_1_1)
        color_to_plot = self.k_off_array
        norm = plt.Normalize(self.k_off_far, self.k_off_near)
        self.animation_patches_lc_1_1 = LineCollection(segments,
                                                       cmap='plasma',
                                                       norm=norm,
                                                       animated=True)
        # Set the values used for colormapping
        self.animation_patches_lc_1_1.set_array(color_to_plot)
        self.animation_patches_lc_1_1.set_linewidth(2)
        self.ax[1, 1].add_collection(self.animation_patches_lc_1_1)
        self.ax[1, 1].set_title('LGN koff')
        ## draw metaphase plate ##
        self.animation_patches_metaphase_plate_line_1_1 = mlines.Line2D([metaphase_plate_1[0],
                                                                         metaphase_plate_2[0]],
                                                                        [metaphase_plate_1[1],
                                                                         metaphase_plate_2[1]],
                                                                        color='black',
                                                                        lw=3,
                                                                        solid_capstyle='round',
                                                                        animated=True)
        self.animation_patches_metaphase_plate_circle_1_1 = Ellipse(xy=(0, 0),
                                                                    width=self.metaphase_plate_length,
                                                                    height=self.metaphase_plate_length,
                                                                    color='black',
                                                                    lw=3,
                                                                    alpha=1.0,
                                                                    animated=True)
        self.ax[1, 1].add_line(self.animation_patches_metaphase_plate_line_1_1)
        self.ax[1, 1].add_patch(self.animation_patches_metaphase_plate_circle_1_1)
        plt.show()
        self.ani = animation.FuncAnimation(self.fig,
                                           func=self.update_cell,
                                           frames=np.arange(1, len(self.time_array)),
                                           init_func=None,
                                           repeat=False,
                                           interval=10,
                                           blit=True)
    def update_cell(self, this_time_index):
        """ update plots of cell shape, to be used for animation """
        #set the current ellipse L and e

        ellipse_L = self.ellipse_L_vect[this_time_index]
        ellipse_e = self.ellipse_e_vect[this_time_index]
        ellipse_length = ellipse_L / np.sqrt(1-np.square(ellipse_e))
        ellipse_height = ellipse_L
        [_, x_array, y_array, _, _, _] = self.get_r_x_y_nx_ny_dldtheta(ellipse_L, ellipse_e)
        spindle_angle = self.spindle_angle_vect[this_time_index]
        #when a solution has been calculated,
        #returns the centrosome and metaphase plate position at time this_time_index
        centrosome_position_1 = self.centrosome_position_1_vect[this_time_index]
        centrosome_position_2 = self.centrosome_position_2_vect[this_time_index]
        metaphase_plate_1 = self.metaphase_plate_1_vect[this_time_index]
        metaphase_plate_2 = self.metaphase_plate_2_vect[this_time_index]
        #draw spindle as a polygon joining the metaphase plate and the centrosome
        Spindle_array = np.array([centrosome_position_1,
                                  metaphase_plate_1,
                                  centrosome_position_2,
                                  metaphase_plate_2])
        #### FIRST PLOT ####
        ## draw cell shape contour ##
        self.animation_patches_cell_shape_contour_0_0.width = 2.0 * ellipse_length
        self.animation_patches_cell_shape_contour_0_0.height = 2.0 * ellipse_height
        ##draw MTs##
        self.animation_patches_MT_1_patch_0_0.center = (centrosome_position_1[0],
                                                        centrosome_position_1[1])
        self.animation_patches_MT_1_patch_0_0.set_clip_path(self.animation_patches_cell_shape_contour_0_0)
        self.animation_patches_MT_2_patch_0_0.center = (centrosome_position_2[0],
                                                        centrosome_position_2[1])
        self.animation_patches_MT_2_patch_0_0.set_clip_path(self.animation_patches_cell_shape_contour_0_0)
        ## draw spindle ##
        self.animation_patches_Spindle_patch_0_0.xy = Spindle_array
        #draw metaphase plate
        if not self.circular_metaphase_plate_vect[this_time_index]:
            self.animation_patches_metaphase_plate_circle_0_0.set_visible(False)
            self.animation_patches_metaphase_plate_line_0_0.set_data([metaphase_plate_1[0],
                                                                      metaphase_plate_2[0]],
                                                                     [metaphase_plate_1[1],
                                                                      metaphase_plate_2[1]])
        else:
            self.animation_patches_metaphase_plate_circle_0_0.set_visible(True)
        #### SECOND PLOT ####
        self.animation_cell_shape_0_1.width = 2.0 * ellipse_length
        self.animation_cell_shape_0_1.height = 2.0 * ellipse_height
#        self.animation_patches_MT_1_patch_0_1.center = (centrosome_position_1[0],
#                                                        centrosome_position_1[1])
#        self.animation_patches_MT_1_patch_0_1.set_clip_path(self.animation_cell_shape_0_1)
#        self.animation_patches_MT_2_patch_0_1.center = (centrosome_position_2[0],
#                                                        centrosome_position_2[1])
#        self.animation_patches_MT_2_patch_0_1.set_clip_path(self.animation_cell_shape_0_1)
        self.animation_patches_Spindle_patch_0_1.xy = Spindle_array
        #draw metaphase plate
        if not self.circular_metaphase_plate_vect[this_time_index]:
            self.animation_patches_metaphase_plate_circle_0_1.set_visible(False)
#            self.animation_patches_MT_1_patch_0_1.set_visible(True)
#            self.animation_patches_MT_2_patch_0_1.set_visible(True)
            self.animation_patches_metaphase_plate_line_0_1.set_data([metaphase_plate_1[0],
                                                                      metaphase_plate_2[0]],
                                                                     [metaphase_plate_1[1],
                                                                      metaphase_plate_2[1]])
        else:
            self.animation_patches_metaphase_plate_circle_0_1.set_visible(True)
#            self.animation_patches_MT_1_patch_0_1.set_visible(False)
#            self.animation_patches_MT_2_patch_0_1.set_visible(False)
        points = np.array([x_array, y_array]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points, np.roll(points, -1, axis=0)], axis=1)
        self.animation_patches_lc.set_segments(segments)
        color_to_plot = np.array(self.sol_c[this_time_index])
        self.animation_patches_lc.set_array(color_to_plot)
        self.ax[0, 1].add_collection(self.animation_patches_lc)
        #### THIRD PLOT ####
        ## draw cell contour ##
        self.animation_cell_shape_1_0.width = 2.0 * ellipse_length
        self.animation_cell_shape_1_0.height = 2.0 * ellipse_height
        ## draw spindle ##
        self.animation_patches_Spindle_patch_1_0.xy = Spindle_array
        self.animation_patches_MT_1_patch_1_0.center = (centrosome_position_1[0],
                                                        centrosome_position_1[1])
        self.animation_patches_MT_1_patch_1_0.set_clip_path(self.animation_cell_shape_1_0)
        self.animation_patches_MT_2_patch_1_0.center = (centrosome_position_2[0],
                                                        centrosome_position_2[1])
        self.animation_patches_MT_2_patch_1_0.set_clip_path(self.animation_cell_shape_1_0)
        self.animation_patches_cell_shape_contour_1_0.width = 2.0 * ellipse_length
        self.animation_patches_cell_shape_contour_1_0.height = 2.0 * ellipse_height
        ## draw metaphase plate ##
        if not self.circular_metaphase_plate_vect[this_time_index]:
            self.animation_patches_Ran_inhibition_patch_1_1_0.set_visible(True)
            self.animation_patches_Ran_inhibition_patch_2_1_0.set_visible(True)
            self.animation_patches_Ran_inhibition_patch_3_1_0.set_visible(True)
            self.animation_patches_Ran_inhibition_patch_circular_1_0.set_visible(False)
            self.animation_patches_metaphase_plate_circle_1_0.set_visible(False)
            self.animation_patches_MT_1_patch_1_0.set_visible(True)
            self.animation_patches_MT_2_patch_1_0.set_visible(True)
            self.animation_patches_metaphase_plate_line_1_0.set_data([metaphase_plate_1[0],
                                                                      metaphase_plate_2[0]],
                                                                     [metaphase_plate_1[1],
                                                                      metaphase_plate_2[1]])
            self.animation_patches_Ran_inhibition_patch_1_1_0.center = (metaphase_plate_1[0],
                                                                        metaphase_plate_1[1])
            self.animation_patches_Ran_inhibition_patch_2_1_0.center = (metaphase_plate_2[0],
                                                                        metaphase_plate_2[1])
            self.animation_patches_Ran_inhibition_patch_3_1_0.set_xy(((self.nucleus_inhibition_range
                                                                       * np.cos(spindle_angle)
                                                                       +self.metaphase_plate_length
                                                                       / 2.0
                                                                       * np.sin(spindle_angle)),
                                                                      (self.nucleus_inhibition_range
                                                                       * np.sin(spindle_angle)
                                                                       -self.metaphase_plate_length
                                                                       / 2.0
                                                                       * np.cos(spindle_angle))))
            self.animation_patches_Ran_inhibition_patch_3_1_0.angle = 180 * (spindle_angle+np.pi/2.0)/np.pi
            self.animation_patches_Ran_inhibition_patch_1_1_0.set_clip_path(self.animation_patches_cell_shape_contour_1_0)
            self.animation_patches_Ran_inhibition_patch_2_1_0.set_clip_path(self.animation_patches_cell_shape_contour_1_0)
            self.animation_patches_Ran_inhibition_patch_3_1_0.set_clip_path(self.animation_patches_cell_shape_contour_1_0)
        else:
            self.animation_patches_Ran_inhibition_patch_1_1_0.set_visible(False)
            self.animation_patches_Ran_inhibition_patch_2_1_0.set_visible(False)
            self.animation_patches_Ran_inhibition_patch_3_1_0.set_visible(False)
            self.animation_patches_Ran_inhibition_patch_circular_1_0.set_visible(True)
            self.animation_patches_Ran_inhibition_patch_circular_1_0.set_clip_path(self.animation_patches_cell_shape_contour_1_0)
            self.animation_patches_MT_1_patch_1_0.set_visible(False)
            self.animation_patches_MT_2_patch_1_0.set_visible(False)
            self.animation_patches_metaphase_plate_circle_1_0.set_visible(True)

        #### FOURTH PLOT ####
        self.animation_patches_cell_shape_contour_1_1.width = 2.0 * ellipse_length
        self.animation_patches_cell_shape_contour_1_1.height = 2.0 * ellipse_height
        ## draw metaphase plate ##
        if not self.circular_metaphase_plate_vect[this_time_index]:
            self.animation_patches_Ran_inhibition_patch_1_1_1.set_visible(True)
            self.animation_patches_Ran_inhibition_patch_2_1_1.set_visible(True)
            self.animation_patches_Ran_inhibition_patch_3_1_1.set_visible(True)
            self.animation_patches_Ran_inhibition_patch_circular_1_1.set_visible(False)
            self.animation_patches_metaphase_plate_circle_1_1.set_visible(False)
            self.animation_patches_metaphase_plate_line_1_1.set_visible(True)
            self.animation_patches_metaphase_plate_line_1_1.set_data([metaphase_plate_1[0],
                                                                      metaphase_plate_2[0]],
                                                                     [metaphase_plate_1[1],
                                                                      metaphase_plate_2[1]])
            self.animation_patches_Ran_inhibition_patch_1_1_1.center = (metaphase_plate_1[0],
                                                                        metaphase_plate_1[1])
            self.animation_patches_Ran_inhibition_patch_2_1_1.center = (metaphase_plate_2[0],
                                                                        metaphase_plate_2[1])
            self.animation_patches_Ran_inhibition_patch_3_1_1.set_xy(((self.nucleus_inhibition_range
                                                                       *np.cos(spindle_angle)
                                                                       +self.metaphase_plate_length/2.0
                                                                       *np.sin(spindle_angle)),
                                                                      (self.nucleus_inhibition_range
                                                                       *np.sin(spindle_angle)
                                                                       -self.metaphase_plate_length/2.0
                                                                       *np.cos(spindle_angle))))
            self.animation_patches_Ran_inhibition_patch_3_1_1.angle = 180.0 * (spindle_angle+np.pi/2.0)/np.pi
            self.animation_patches_Ran_inhibition_patch_1_1_1.set_clip_path(self.animation_patches_cell_shape_contour_1_1)
            self.animation_patches_Ran_inhibition_patch_2_1_1.set_clip_path(self.animation_patches_cell_shape_contour_1_1)
            self.animation_patches_Ran_inhibition_patch_3_1_1.set_clip_path(self.animation_patches_cell_shape_contour_1_1)
        else:
            self.animation_patches_Ran_inhibition_patch_1_1_1.set_visible(False)
            self.animation_patches_Ran_inhibition_patch_2_1_1.set_visible(False)
            self.animation_patches_Ran_inhibition_patch_3_1_1.set_visible(False)
            self.animation_patches_Ran_inhibition_patch_circular_1_1.set_visible(True)
            self.animation_patches_Ran_inhibition_patch_circular_1_1.set_clip_path(self.animation_patches_cell_shape_contour_1_1)
            self.animation_patches_metaphase_plate_line_1_1.set_visible(False)
            self.animation_patches_metaphase_plate_circle_1_1.set_visible(True)
        points = np.array([x_array, y_array]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points, np.roll(points, -1, axis=0)], axis=1)
        self.animation_patches_lc_1_1.set_segments(segments)
        color_to_plot = self.k_off_array_vect[this_time_index]
        self.animation_patches_lc_1_1.set_array(color_to_plot)
        return [self.animation_patches_cell_shape_contour_0_0,
                self.animation_patches_MT_1_patch_0_0,
                self.animation_patches_MT_2_patch_0_0,
                self.animation_patches_Spindle_patch_0_0,
                self.animation_patches_metaphase_plate_line_0_0,
                self.animation_patches_metaphase_plate_circle_0_0,
                self.animation_cell_shape_0_1,
                #self.animation_patches_MT_1_patch_0_1,
                #self.animation_patches_MT_2_patch_0_1,
                self.animation_patches_Spindle_patch_0_1,
                self.animation_patches_metaphase_plate_line_0_1,
                self.animation_patches_metaphase_plate_circle_0_1,
                self.animation_patches_lc,
                self.animation_cell_shape_1_0,
                self.animation_patches_MT_1_patch_1_0,
                self.animation_patches_MT_2_patch_1_0,
                self.animation_patches_Spindle_patch_1_0,
                self.animation_patches_Ran_inhibition_patch_1_1_0,
                self.animation_patches_Ran_inhibition_patch_2_1_0,
                self.animation_patches_Ran_inhibition_patch_3_1_0,
                self.animation_patches_Ran_inhibition_patch_circular_1_0,
                self.animation_patches_metaphase_plate_line_1_0,
                self.animation_patches_metaphase_plate_circle_1_0,
                self.animation_patches_cell_shape_contour_1_0,
                self.animation_patches_Ran_inhibition_patch_1_1_1,
                self.animation_patches_Ran_inhibition_patch_3_1_1,
                self.animation_patches_Ran_inhibition_patch_2_1_1,
                self.animation_patches_Ran_inhibition_patch_circular_1_1,
                self.animation_patches_metaphase_plate_line_1_1,
                self.animation_patches_metaphase_plate_circle_1_1,
                self.animation_patches_cell_shape_contour_1_1,
                self.animation_patches_lc_1_1]
# =============================================================================
#  ADDITIONAL USEFUL FUNCTIONS
# =============================================================================
# from: https://gist.github.com/nim65s/5e9902cd67f094ce65b0
def distance_segment_point(A, B, P):
    """ segment line AB, point P, where each one is an array([x, y])
    return the distance of the point P to the segment AB"""
    if all(A == P) or all(B == P): #"all" test here both coordinates
        return 0
    projection_1 = (np.dot((P - A), (B - A))
                    / np.linalg.norm(P - A)
                    / np.linalg.norm(B - A))
    #This is done here to avoid numerical imprecisions
    #returning a result slightly outside [-1,1]
    projection_1 = np.minimum(projection_1, 1)
    projection_1 = np.maximum(projection_1, -1)
    #if the point is in the half-space below A, orthogonal to AB
    if projection_1 <= 0:
        return np.linalg.norm(P - A)
    projection_2 = (np.dot((P - B), (A - B))
                    / np.linalg.norm(P - B)
                    / np.linalg.norm(A - B))
    #This is done here to avoid numerical imprecisions
    #returning a result slightly outside [-1,1]
    projection_2 = np.minimum(projection_2, 1)
    projection_2 = np.maximum(projection_2, -1)
    #if the point is in the half-space above B, orthogonal to AB
    if projection_2 <= 0:
        return np.linalg.norm(P - B)
    #if the point is within the stripe between the two half-spaces
    return np.linalg.norm(np.cross(A - B, A - P)) / np.linalg.norm(B - A)
def distance_circle_point(O, R, P):
    """ return distance between the point P and the circle
    with center in O and radius R
    """
    distance = np.linalg.norm(P-O) - R
    if distance < 0:
        return 0
    return distance
