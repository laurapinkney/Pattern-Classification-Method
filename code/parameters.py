# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 10:31:04 2023

@author: mm18lp
"""
# VERSION 2.2


# Parameters Python file

import numpy as np 
import math
import os

###############################################################################
                        # Running on HPC/local device    

aire_run = True # True corresponds to running a task array on HPC, running a 
                # series of independent simulations for different values of \mu and \nu 

###############################################################################
                        # mu/nu domain size
                        
full_run = True    # full 270 degree wedge

chaos_wedge = False   # smaller 45-135 degrees wedge

      
###############################################################################
                          # PDE parameters
                          
q = 1/np.sqrt(7)    # critical wavenumber

sigma_0 = -2     # growth rate of k=0 mode


Q1 = -1.4      # u^2 coefficient

Q2 = -2.75        # ulapu coefficient

Q3 = -3.5      # |grad u|^2 coefficent

C1 = -2.75       # u^3 coefficient

C2 = -7.75         # u^2 lapu coefficient

C3 = -16.5       # u |grad u|^2 coefficient


###############################################################################
                    # Domain Size and Lattice indexing
                           
rhombic_lattice = False     # fits k1, k2 and q1 exactly

hexagonal_lattice = True   # fits q1, q2 and q3 exactly

small_box = False            # Fits one wavelength, either rhombic or hexagonal

large_box = True           # Large approx square box, either rhombic or hexagonal    

###############################################################################
                       # Initial Condition

preset_random_initial = True # import preset random initial condition

new_random_initial = False   # construct new random initial condition - helpful when changing box size

longer_run_intial = False  # continue an existing run for longer time

rerun_initial = False # Rerun simulation from start - importing saved U_ic
                     
small_to_large_box = False # Use initial condition from small box to run in larger box

parameter_continuation_initial = False  #  use initial condition from prior run with different parameters


if parameter_continuation_initial:
    directory = 'simruns1'   #  path to import initial condition from
    
###############################################################################
                        # Timestepping Parameters

if longer_run_intial:
    t = np.loadtxt(r'dataoutput/t_vals.txt')[-1]        # set t to be final value from previous run

else:
    t= 0 

save_step = 400     # frequency data is saved

h = 0.15            # timestep

number_of_steps = 80000  # number of timesteps

break_t_equilibrium = t+5000  # earliest possible time to mark converged to equilibria

save_U_ic = 1000     # frequency U_n is saved, to be used as new initial condition in case of job failure

convergence_tol = 1.0e-9   # To stop code when converged to equilibrium

k_filtering_radius = 3.5   # Filtering modes with wavenumber larger than this

complex_tol = '%.7e' # number of dp when saving complex data

average_cutoff_percentage = 20     # first x% of values cut off when computing average of criteria 2 and 3

average_cutoff = number_of_steps//(save_step)*average_cutoff_percentage//100

###############################################################################

    
if aire_run:      # import parameter values from text file with correct index from task array. Using SLURM notation

    if full_run:     # 689 runs 270 degree wedge

        paramlist = np.loadtxt('../../pythonfiles/paramlist.txt')       # text file with list of (mu,nu) values
        
        index = int(os.environ.get('SLURM_ARRAY_TASK_ID',1))    
        
        [r,chi] = paramlist[index-1]        #  [ length of (mu,nu)  ,  angle betweeen mu and nu, i.e. polar co-ordinates ]

        
    elif chaos_wedge:   # 234 runs, only considering 45-135 degree wedge (chaos search)
        
        paramlist = np.loadtxt('../../pythonfiles/paramlistwedge.txt')      # text file with different (mu,nu) values

        index = int(os.environ.get('SLURM_ARRAY_TASK_ID',1))
        
        [r,chi] = paramlist[index-1]        #  [ length of (mu,nu)  ,  angle betweeen mu and nu, i.e. polar co-ordinates ]

        
else:
    
    r = 0.025               # set yourself if running locally
    
    chi = 150
    

###############################################################################
                        # Plotting Parameters

annulus_interval_width = 0.075  # width of annulus for critical circles used when plotting

number_of_wavelengths_x_small = 3       # number of pattern repeats for small boxes used to determine local patterns

bin_width = 5   # angular width of bins within annulus

###############################################################################
                        # PDE Type and Dealiasing
                        
filtering = True    # Dealiasing

linear = False      # PDE with only linear terms

no_spatial = False  # PDE with Q1u^2 and C1u^3 as only nonlinearities 

no_cubic_spatial = False # PDE with all quadratic nonlinearities, only C1u^3 for cubic terms

full_nonlinear = True    # Full PDE with all nonlinearities 

###############################################################################
###############################################################################
###############################################################################

# PDE parameters determined from above 

thetaz = 2*math.acos(q/2) # angle between k1 and k2

mu = r*math.cos(chi*math.pi/180) 

nu = r*math.sin(chi*math.pi/180)

###############################################################################

# Setting up Domain

if rhombic_lattice:
    
    aspect_ratio_x = 1/math.cos(thetaz/2)
    
    aspect_ratio_y = 1/math.sin(thetaz/2)
    
elif hexagonal_lattice:     # k=q hexagonal lattice
    
    aspect_ratio_x = 2/q

    aspect_ratio_y = 2/(q*np.sqrt(3))


if small_box:                                                            # small box 
    
    number_of_wavelengths_x = 1

    number_of_wavelengths_y = 1
    
    Nx = 64                                              # Number of Fourier Modes in x-direction
    
    Ny = 64                                              # Number of Fourier Modes in y-direction
    
    
elif large_box:
    if rhombic_lattice:
        ratio = round(aspect_ratio_x/aspect_ratio_y)                               # 1:5 ratio approx 

        number_of_wavelengths_x = 6                      # lots of small boxes joined together. 6 gives approx 30x30 wavelength domain
        
        number_of_wavelengths_y = 6 * ratio                # makes box approx square
        
        Nx = 256                                        # Number of Fourier Modes in x-direction
        
        Ny = 256                                        # Number of Fourier Modes in y-direction
        
    elif hexagonal_lattice:
        
        number_of_wavelengths_x =  4 * 2                  # sqrt(3) ratio is approx 5/3 or 7/4 (better)           
        
        number_of_wavelengths_y =  7 * 2                  # to make box more square

        Nx = 384                                         # Number of Fourier Modes in x-direction
        
        Ny = 384                                         # Number of Fourier Modes in y-direction
        
        
    
Lx = aspect_ratio_x * number_of_wavelengths_x * 2 * math.pi                 # box size x-direction

Ly = aspect_ratio_y * number_of_wavelengths_y * 2 * math.pi                 # box size y-direction
        

x = np.arange(0.0, Nx)*(Lx/(Nx))                        # gridpoints x-direction

y = np.arange(0.0, Ny)*(Ly/(Ny))                        # grid points y-direction

X, Y = np.meshgrid(x, y, indexing='ij')                 # constructing 2D mesh

###############################################################################
    
                                # wavevectors

k1 = [q/2,-np.sqrt(1-q**2/4)]

k2 = [q/2,np.sqrt(1-q**2/4)]

q1 = [k1[0]+k2[0],k1[1]+k2[1]]

q2 = [-q/2,-np.sqrt(3)*q/2]

q3 = [-q/2,np.sqrt(3)*q/2]

k3 = [-q/4-np.sqrt(3)/2*np.sqrt(1-q**2/4),-np.sqrt(3)*q/4+1/2*np.sqrt(1-q**2/4)]

k4 = [q2[0]-k3[0],q2[1]-k3[1]]

k5 = [-k1[0]-k3[0],-k1[1]-k3[1]]

k6 = [q3[0]-k5[0],q3[1]-k5[1]]


###############################################################################

                #Quadratic and Cubic Parameters in ODEs
                
                # Determined with Weakly Nonlinear Theory

delta_z   = (Q2+Q3-Q1)*q**4/(9*sigma_0*(q**2-4)**2)
        
delta_w   = (q**2*(Q2+Q3)-Q1)/(9*sigma_0*(4*q**2-1)**2)
        
beta_a    = (2*(Q2-Q1-Q3*np.cos(thetaz))*q**4)/(sigma_0*(4*np.sin(thetaz/2)**2-1)**2*(4*np.sin(thetaz/2)**2-q**2)**2)
        
beta_60   = (q**4*(2*Q2+Q3-2*Q1))/(4*sigma_0*(q**2-3)**2)

beta_60ma = 8*q**2*(4*Q2-4*Q1+Q3*(q**2-2-np.sqrt(3)*q*np.sqrt(4-q**2)))/(sigma_0*(np.sqrt(3)*np.sqrt(4-q**2)-q)**2*(q**2+np.sqrt(3)*q*np.sqrt(4-q**2)-2)**2)

phi_30ma  = 8*q**2*(2*(1+q**2)*Q2-4*Q1-Q3*(q**2+np.sqrt(3)*q*np.sqrt(4-q**2)))/(sigma_0*(np.sqrt(3)*np.sqrt(4-q**2)-q)**2*(q**2+np.sqrt(3)*q*np.sqrt(4-q**2)-2)**2)

beta_60pa = 8*q**2*(4*Q2-4*Q1+Q3*(q**2-2+np.sqrt(3)*q*np.sqrt(4-q**2)))/(sigma_0*(np.sqrt(3)*np.sqrt(4-q**2)+q)**2*(-q**2+np.sqrt(3)*q*np.sqrt(4-q**2)+2)**2)

phi_30pa  = 8*q**2*(2*(1+q**2)*Q2-4*Q1-Q3*(q**2-np.sqrt(3)*q*np.sqrt(4-q**2)))/(sigma_0*(np.sqrt(3)*np.sqrt(4-q**2)+q)**2*(-q**2+np.sqrt(3)*q*np.sqrt(4-q**2)+2)**2)

chi_60ma  = 8*q**4*(4*Q2-4*Q1+Q3*(-q**2+2+np.sqrt(3)*q*np.sqrt(4-q**2)))/(sigma_0*(np.sqrt(3)*q*np.sqrt(4-q**2)-q**2+4)**2*(-3*q**2+np.sqrt(3)*q*np.sqrt(4-q**2)+6)**2)

chi_60pa  = 8*q**4*(4*Q2-4*Q1+Q3*(-q**2+2-np.sqrt(3)*q*np.sqrt(4-q**2)))/(sigma_0*(np.sqrt(3)*q*np.sqrt(4-q**2)+q**2-4)**2*(3*q**2+np.sqrt(3)*q*np.sqrt(4-q**2)-6)**2)

psi_30ma  = 8*q**2*(2*(1+q**2)*Q2-4*Q1+Q3*(q**2+np.sqrt(3)*q*np.sqrt(4-q**2)))/(sigma_0*(np.sqrt(3)*np.sqrt(4-q**2)+3*q)**2*(q**2+np.sqrt(3)*q*np.sqrt(4-q**2)+2)**2)

psi_30pa  = 8*q**2*(2*(1+q**2)*Q2-4*Q1+Q3*(q**2-np.sqrt(3)*q*np.sqrt(4-q**2)))/(sigma_0*(np.sqrt(3)*np.sqrt(4-q**2)-3*q)**2*(-q**2+np.sqrt(3)*q*np.sqrt(4-q**2)-2)**2)

psi_90ma  = (-2*Q1+Q2*(1+q**2)+q**2*Q3)/(4*sigma_0*(q**2+1)**2)

psi_60    = (2*q**2*Q2-2*Q1+q**2*Q3)/(4*sigma_0*(1-3*q**2)**2)
        
gamma_z = 2*(Q2-Q3-Q1)/sigma_0
        
gamma_w = 2*(q**2*Q2-q**2*Q3-Q1)/sigma_0

# ODE parameters
     
Qzw = 2*Q1 -(1+q**2)*Q2+q**2*Q3
 
Qzz = 2*Q1 -2*Q2 - 2*Q3*math.cos(thetaz)

Qzhex = 2*Q1 - 2*Q2 + Q3

Qwhex = 2*Q1 - 2*q**2*Q2 + q**2*Q3

A = 2*Q1*(delta_z+gamma_z)-Q2*(5*delta_z+gamma_z)+4*Q3*delta_z+3*C1 -3*C2 + C3   

B_a = 2*Q1*(beta_a + gamma_z)-Q2*(beta_a*(5-q**2)+gamma_z)-Q3*beta_a*(q**2-4)+6*C1 -6*C2 + 2*C3   # B_\alpha 

B_60 = 2*Q1*(beta_60 + gamma_z)-Q2*(4*beta_60+gamma_z)+3*Q3*beta_60 +6*C1 -6*C2 + 2*C3 

B_60ma = (2*Q1*(beta_60ma + chi_60ma + gamma_z)-Q2*(beta_60ma*(2+q**2/2 - np.sqrt(3)*q*np.sqrt(4-q**2)/2) +               # B_{60-\alpha}
                                                 chi_60ma *(4-q**2/2 + np.sqrt(3)*q*np.sqrt(4-q**2)/2) +gamma_z)
        +Q3*(beta_60ma*(1+q**2/2 - np.sqrt(3)*q*np.sqrt(4-q**2)/2) + chi_60ma * (3-q**2/2 + np.sqrt(3)*q*np.sqrt(4-q**2)/2))
        +6*C1 -6*C2 + 2*C3)

B_60pa = (2*Q1*(beta_60pa + chi_60pa + gamma_z)-Q2*(beta_60pa*(2+q**2/2 + np.sqrt(3)*q*np.sqrt(4-q**2)/2) +               # B_{60+\alpha}
                                                 chi_60pa *(4-q**2/2 - np.sqrt(3)*q*np.sqrt(4-q**2)/2) +gamma_z)
        +Q3*(beta_60pa*(1+q**2/2 + np.sqrt(3)*q*np.sqrt(4-q**2)/2) + chi_60pa * (3-q**2/2 - np.sqrt(3)*q*np.sqrt(4-q**2)/2)) 
        +6*C1 -6*C2 + 2*C3)

C_90ma = 2*Q1*(psi_90ma+gamma_w)-Q2*(psi_90ma*(1+3*q**2)+gamma_w) +3*q**2*Q3*psi_90ma + 6*C1 -(2+4*q**2)*C2 + 2*q**2*C3     # C_{90-\alpha/2} 

C_30pa = (2*Q1*(psi_30pa + phi_30pa + gamma_w)-Q2*(psi_30pa*(1+5*q**2/2 - np.sqrt(3)*q*np.sqrt(4-q**2)/2) +                # C_{30+\alpha/2}
                                                 phi_30pa *(1+3*q**2/2 + np.sqrt(3)*q*np.sqrt(4-q**2)/2) +gamma_w)
        +Q3*(psi_30pa*(5*q**2/2 - np.sqrt(3)*q*np.sqrt(4-q**2)/2) + phi_30pa * (3*q**2/2 + np.sqrt(3)*q*np.sqrt(4-q**2)/2))  
        +6*C1 -(2+4*q**2)*C2 + 2*q**2*C3)

C_30ma = (2*Q1*(psi_30ma + phi_30ma + gamma_w)-Q2*(psi_30ma*(1+5*q**2/2 + np.sqrt(3)*q*np.sqrt(4-q**2)/2) +                # C_{30-\alpha/2}
                                                 phi_30ma *(1+3*q**2/2 - np.sqrt(3)*q*np.sqrt(4-q**2)/2) +gamma_w)
        +Q3*(psi_30ma*(5*q**2/2 + np.sqrt(3)*q*np.sqrt(4-q**2)/2) + phi_30ma * (3*q**2/2 - np.sqrt(3)*q*np.sqrt(4-q**2)/2)) 
        +6*C1 -(2+4*q**2)*C2 + 2*q**2*C3)

K1p = (2*Q1*(phi_30ma + phi_30pa)-Q2*(phi_30ma*(2+q**2/2 - np.sqrt(3)*q*np.sqrt(4-q**2)/2) +                                 # K1
                                                 phi_30pa *(2+q**2/2 + np.sqrt(3)*q*np.sqrt(4-q**2)/2))                     
        +Q3*(phi_30ma*(1+q**2/2 - np.sqrt(3)*q*np.sqrt(4-q**2)/2) + phi_30pa * (1+q**2/2 + np.sqrt(3)*q*np.sqrt(4-q**2)/2)) #
        +6*C1 -(4+2*q**2)*C2 + (1+q**2)*C3)

K2p = (2*Q1*(phi_30ma + phi_30pa)-Q2*(phi_30ma*(1+3*q**2/2 - np.sqrt(3)*q*np.sqrt(4-q**2)/2) +                               # K2
                                                 phi_30pa *(1+3*q**2/2 + np.sqrt(3)*q*np.sqrt(4-q**2)/2))
        +Q3*(phi_30ma*(3*q**2/2 - np.sqrt(3)*q*np.sqrt(4-q**2)/2) + phi_30pa * (3*q**2/2 + np.sqrt(3)*q*np.sqrt(4-q**2)/2)) # using K2p instead of K2 to not confuse with wavenumber mesh
        +6*C1 -(2+4*q**2)*C2 + 2*q**2*C3)

K3p = (2*Q1*(beta_60pa + phi_30ma)-Q2*(beta_60pa*(1+3*q**2/2 + np.sqrt(3)*q*np.sqrt(4-q**2)/2) +                             # K3          
                                                 phi_30ma *(2+q**2/2 - np.sqrt(3)*q*np.sqrt(4-q**2)/2))
        +Q3*(beta_60pa*(3*q**2/2 + np.sqrt(3)*q*np.sqrt(4-q**2)/2) + phi_30ma * (1+q**2/2 - np.sqrt(3)*q*np.sqrt(4-q**2)/2)) # 
        +6*C1 -(4+2*q**2)*C2 + (1+q**2)*C3)

K4p = (2*Q1*(beta_60ma + phi_30pa)-Q2*(beta_60ma*(1+3*q**2/2 - np.sqrt(3)*q*np.sqrt(4-q**2)/2) +                             # K4   
                                                 phi_30pa *(2+q**2/2 + np.sqrt(3)*q*np.sqrt(4-q**2)/2))
        +Q3*(beta_60ma*(3*q**2/2 - np.sqrt(3)*q*np.sqrt(4-q**2)/2) + phi_30pa * (1+q**2/2 + np.sqrt(3)*q*np.sqrt(4-q**2)/2)) # 
        +6*C1 -(4+2*q**2)*C2 + (1+q**2)*C3)


D = 2*Q1*(delta_w+gamma_w)-q**2*Q2*(5*delta_w+gamma_w)+4*q**2*Q3*delta_w+3*C1 -3*q**2*C2+q**2*C3                             # D

E_90ma = 2*Q1*(psi_90ma+gamma_z)-Q2*(2*(1+q**2)*psi_90ma+q**2*gamma_z)+Q3*(2+q**2)*psi_90ma+6*C1 -(2*q**2+4)*C2 +2*C3        # E_{90-\alpha/2}

E_30ma = (2*Q1*(psi_30ma + phi_30ma + gamma_z)-Q2*(psi_30ma*(2+3*q**2/2 + np.sqrt(3)*q*np.sqrt(4-q**2)/2) +                # E_{30-\alpha/2}
                                                 phi_30ma *(2+q**2/2 - np.sqrt(3)*q*np.sqrt(4-q**2)/2) +q**2*gamma_z)
        +Q3*(psi_30ma*(2+q**2/2 + np.sqrt(3)*q*np.sqrt(4-q**2)/2) + phi_30ma * (2-q**2/2 - np.sqrt(3)*q*np.sqrt(4-q**2)/2))
        +6*C1 -(4+2*q**2)*C2 + 2*C3)

E_30pa = (2*Q1*(psi_30pa + phi_30pa + gamma_z)-Q2*(psi_30pa*(2+3*q**2/2 - np.sqrt(3)*q*np.sqrt(4-q**2)/2) +                # E_{30+\alpha/2}
                                                 phi_30pa *(2+q**2/2 + np.sqrt(3)*q*np.sqrt(4-q**2)/2) +q**2*gamma_z)
        +Q3*(psi_30pa*(2+q**2/2 - np.sqrt(3)*q*np.sqrt(4-q**2)/2) + phi_30pa * (2-q**2/2 + np.sqrt(3)*q*np.sqrt(4-q**2)/2)) 
        +6*C1 -(4+2*q**2)*C2 + 2*C3)


F_60 = 2*Q1*(psi_60 + gamma_w)-Q2*(4*q**2*psi_60+q**2*gamma_w)+3*Q3*q**2*psi_60 +6*C1 -6*q**2*C2 + 2*q**2*C3                # F_{60}


Lhex = (2*Q1*(beta_60pa + beta_60ma)-Q2*(beta_60pa*(2+q**2/2 + np.sqrt(3)*q*np.sqrt(4-q**2)/2) +               
                                                 beta_60ma *(2+q**2/2 - np.sqrt(3)*q*np.sqrt(4-q**2)/2))                        # L_{hex}
        +Q3*(beta_60pa*(2-q**2/2 + np.sqrt(3)*q*np.sqrt(4-q**2)/2) + beta_60ma * (2-q**2/2 - np.sqrt(3)*q*np.sqrt(4-q**2)/2))
        +6*C1 -6*C2 + (3-q**2)*C3)

L1 = (2*Q1*(phi_30ma + phi_30pa)-Q2*(phi_30ma*(2+q**2/2 - np.sqrt(3)*q*np.sqrt(4-q**2)/2) +                                     # L_1
                                                 phi_30pa *(2+q**2/2 + np.sqrt(3)*q*np.sqrt(4-q**2)/2))  
        +Q3*(phi_30ma*(2-q**2/2 - np.sqrt(3)*q*np.sqrt(4-q**2)/2) + phi_30pa * (2-q**2/2 + np.sqrt(3)*q*np.sqrt(4-q**2)/2))
        +6*C1 -(4+2*q**2)*C2 + 2*C3)


if q == 1/np.sqrt(7):           # extra terms for q=1/sqrt(7) only
    
    Kzww = (2*Q1*(psi_30pa + psi_60 + psi_90ma)-Q2*(psi_30pa*(1+5*q**2/2 - np.sqrt(3)*q*np.sqrt(4-q**2)/2) +
                                                     psi_60 *(1+3*q**2) + psi_90ma*(1+3*q**2))
            +Q3*(-2*q**2*psi_30pa + psi_60*(-3*q**2/2 + np.sqrt(3)*q*np.sqrt(4-q**2)/2) 
                 + psi_90ma * (-3*q**2/2 + np.sqrt(3)*q*np.sqrt(4-q**2)/2))
            +6*C1 -(2+4*q**2)*C2 - C3*(5*q**2/2 - np.sqrt(3)*q*np.sqrt(4-q**2)))                                                        # K_{zww}

    Kww = 2*Q1*(delta_w+psi_60)-Q2*(5*q**2*delta_w+4*q**2*psi_60)-Q3*(2*q**2*delta_w + 3*q**2*psi_60) +3*C1 -3*q**2*C2 - 2*q**2*C3      # K_{ww}

    
    Lwz = (2*Q1*(delta_w+phi_30ma)-Q2*(delta_w*(4*q**2+1)+ phi_30ma *(1+3*q**2/2 - np.sqrt(3)*q*np.sqrt(4-q**2)/2) )                    # L_{wz}
            +Q3*(delta_w*(q**2+np.sqrt(3)*q*np.sqrt(4-q**2)) + phi_30ma *(-3*q**2/2 + np.sqrt(3)*q*np.sqrt(4-q**2)/2))
            +3*C1 -(2*q**2+1)*C2+C3*(-q**2/2+np.sqrt(3)*q*np.sqrt(4-q**2)/2))
     
    Lwzz =  (2*Q1*(beta_60ma +psi_30pa + psi_90ma)-Q2*(beta_60ma*(1+3*q**2/2 - np.sqrt(3)*q*np.sqrt(4-q**2)/2)                          # L_{wzz}
                                                       + psi_30pa*(2+3*q**2/2 - np.sqrt(3)*q*np.sqrt(4-q**2)/2) + psi_90ma*(2+2*q**2))
            +Q3*(beta_60ma*(-3*q**2/2 + np.sqrt(3)*q*np.sqrt(4-q**2)/2) + psi_30pa*(1-3*q**2/2 + np.sqrt(3)*q*np.sqrt(4-q**2)/2) 
                 + psi_90ma * (1-q**2 + np.sqrt(3)*q*np.sqrt(4-q**2)))
            +6*C1 -(4+2*q**2)*C2 - C3*(1-2*q**2 - np.sqrt(3)*q*np.sqrt(4-q**2)))

    Lwwz = (2*Q1*(phi_30ma + psi_30pa + psi_60)-Q2*(phi_30ma*(1+3*q**2/2 - np.sqrt(3)*q*np.sqrt(4-q**2)/2)                              # L_{wwz}
                                                    + psi_30pa*(1+5*q**2/2 - np.sqrt(3)*q*np.sqrt(4-q**2)/2) + psi_60 *(3*q**2+1) )
            +Q3*( phi_30ma * (-3*q**2/2 + np.sqrt(3)*q*np.sqrt(4-q**2)/2) +psi_30pa*(-q**2/2 + np.sqrt(3)*q*np.sqrt(4-q**2)/2)
                 + psi_60*np.sqrt(3)*q*np.sqrt(4-q**2) )
            +6*C1 -(2+4*q**2)*C2 + C3*(-q**2+ np.sqrt(3)*q*np.sqrt(4-q**2)))
    
else:           # terms do not exist in ODEs when q \neq 1/\sqrt{7}
    Kzww = 0
    
    Kww = 0 
    
    Lwzz = 0 
    
    Lwwz = 0 
    
    Lwz = 0

    
print(['r=',r],['chi=',chi],['Qzw=', Qzw],['Qzz=',Qzz], ['Qzhex=',Qzhex], ['Qwhex=',Qwhex],['A=',A],
      ['B_a=',B_a],['B_60=',B_60],['B_60ma=',B_60ma],['B_60pa=',B_60pa],['C_90ma=',C_90ma],
      ['C_30ma=',C_30ma], ['C_30pa=',C_30pa],['K1=',K1p],['K2=',K2p],['K3=',K3p],['K4=',K4p],
      ['Kzww=',Kzww], ['Kww=',Kww],['D=',D],['E_90ma=',E_90ma],['E_30ma=',E_30ma],['E_30pa=',E_30pa],
      ['F_60=',F_60],['Lhex=',Lhex],['L1=',L1],['Lwz=',Lwz],['Lwzz=',Lwzz],['Lwwz=',Lwwz],sep=os.linesep)




###############################################################################
                    # create a list of wavenumbers

# first set up the integer wavenumbers
# x component of wavevector
kx = np.arange(0, Nx) * 1.0
for i in range(int(Nx/2+1), Nx):
    kx[i] = i-Nx

# y component of wavevector
ky = np.arange(0, Ny) * 1.0
for i in range(int(Ny/2+1), Ny):
    ky[i] = i-Ny


# Scale integers to physical wavenumbers

kx = kx*2*math.pi/Lx

ky = ky*2*math.pi/Ly

# Squared wavevectors
kx2 = np.square(kx)

ky2 = np.square(ky)


# Ensures correct dimensions on wavevectors
KX, KY = np.meshgrid(kx, ky, indexing='ij')

KX2, KY2 = np.meshgrid(kx2, ky2, indexing='ij')  # Matricies for kx^2 and ky^2


K2 = KX2 + KY2  # kx^2 + ky^2 = k^2

K4 = np.square(K2)          # k^4

K6 = np.multiply(K2, K4)    # k^6

K8 = np.square(K4)          # k^8

K = np.sqrt(K2)             # k

###############################################################################

# Initial condition

if new_random_initial:
    U_ic = 1e-1*np.random.random((Nx,Ny))

elif preset_random_initial:                                                                          # same initial condition for all runs
    U_ic = np.loadtxt('../../pythonfiles/random_U_ic.txt')

elif parameter_continuation_initial:                                                                                 # previous Q value same mu nu initial condition
    U_ic = np.loadtxt(f'../../../{directory}/run_output/run-{index}/dataoutput/new_U_ic.txt')
    
elif longer_run_intial:
    U_ic = np.loadtxt(r'dataoutput/new_U_ic.txt')

elif rerun_initial:
    U_ic = np.loadtxt(r'dataoutput/U_ic.txt')
    

elif small_to_large_box:
    small_U_ic = np.loadtxt(r'dataoutput/new_U_ic.txt')  # importing previous data as inital condition
    
    old = small_U_ic.shape
    
    new = (Nx,Ny)
    
    U_ic = np.tile(small_U_ic,(int(new[0]/old[0]),int(new[1]/old[1]))) +1e-10*np.random.random((Nx,Ny))
    


np.savetxt(r'dataoutput/U_ic.txt',U_ic)                                         # save initial condition
    

###############################################################################

if aire_run:
    if index ==1:   # saving mesh information on first simulation only
    
        np.savetxt(r'dataoutput/X.txt', X)
        np.savetxt(r'dataoutput/Y.txt', Y)
        np.savetxt(r'dataoutput/KX.txt', KX)
        np.savetxt(r'dataoutput/KY.txt', KY)

###############################################################################
                   # Printing information to text file

filelocation = 'parameters_data.txt'

with open(filelocation,'w') as outfile:
    outfile.write('# Initial Data for ETD4RK iteration \n \n')
    
    outfile.write(' Q1= {0}\n'.format(Q1))
    outfile.write(' Q2= {0}\n'.format(Q2))
    outfile.write(' Q3= {0}\n'.format(Q3))
    outfile.write(' C1= {0}\n'.format(C1))
    outfile.write(' C2= {0}\n'.format(C2))
    outfile.write(' C3= {0}\n \n'.format(C3))

    outfile.write(' q = {0}\n'.format(q))
    outfile.write(' sigma_0 = {0}\n \n'.format(sigma_0))
    
    outfile.write(' r = {0}\n'.format(r))
    outfile.write(' chi = {0}\n \n'.format(chi))
    
    outfile.write(' mu = {0}\n'.format(mu))
    outfile.write(' nu = {0}\n \n'.format(nu))
    
    outfile.write(' Nx = {0}\n'.format(Nx))
    outfile.write(' Ny = {0}\n \n'.format(Ny))
    
    outfile.write(' Lx = {0}\n'.format(Lx))
    outfile.write(' Ly = {0}\n \n'.format(Ly))
    
    outfile.write(' save_step = {0}\n'.format(save_step))
    outfile.write(' timestep = {0}\n'.format(h))
    outfile.write(' number_of_steps = {0}\n'.format(number_of_steps))
    outfile.write(' convergence_tol = {0}\n'.format(convergence_tol))
    outfile.write(' fourier_filtering_radius = {0}\n'.format(k_filtering_radius))
    outfile.write(' complex_data_tol = {0}\n \n'.format(complex_tol))

    outfile.write(' Qzw = {0}\n'.format(Qzw))
    outfile.write(' Qzz = {0}\n'.format(Qzz))
    outfile.write(' Qzhex = {0}\n'.format(Qzhex))
    outfile.write(' Qwhex = {0}\n'.format(Qwhex))
    outfile.write(' A = {0}\n'.format(A))
    outfile.write(' B_a = {0}\n'.format(B_a))
    outfile.write(' B_60 = {0}\n'.format(B_60))
    outfile.write(' B_60ma = {0}\n'.format(B_60ma))
    outfile.write(' B_60pa = {0}\n'.format(B_60pa))
    outfile.write(' C_90ma = {0}\n'.format(C_90ma))
    outfile.write(' C_30ma = {0}\n'.format(C_30ma))
    outfile.write(' C_30pa = {0}\n'.format(C_30pa))
    outfile.write(' K1 = {0}\n'.format(K1p))
    outfile.write(' K2 = {0}\n'.format(K2p))
    outfile.write(' K3 = {0}\n'.format(K3p))
    outfile.write(' K4 = {0}\n'.format(K4p))
    outfile.write(' Kzww = {0}\n'.format(Kzww))
    outfile.write(' Kww = {0}\n'.format(Kww))
    outfile.write(' D = {0}\n'.format(D))
    outfile.write(' E_90ma = {0}\n'.format(E_90ma))
    outfile.write(' E_30ma = {0}\n'.format(E_30ma))
    outfile.write(' E_30pa = {0}\n'.format(E_30pa))
    outfile.write(' F_60 = {0}\n'.format(F_60))
    outfile.write(' Lhex = {0}\n'.format(Lhex))
    outfile.write(' L1 = {0}\n'.format(L1))
    outfile.write(' Lwz = {0}\n'.format(Lwz))
    outfile.write(' Lwzz = {0}\n'.format(Lwzz))
    outfile.write(' Lwwz = {0}\n'.format(Lwwz))
    
    
    




