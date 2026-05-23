# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 10:31:06 2023

@author: mm18lp
"""

# VERSION 2.2

import numpy as np
import os
import math

from scipy.fft import fftn, ifftn                   # was using scipy.fftpack   

import parameters as param

import functions as fn
                                                    # add if statement for arc/local

###############################################################################

# Importing Parameters 

Q1, Q2, Q3, C1, C2, C3 = param.Q1, param.Q2, param.Q3, param.C1, param.C2, param.C3

r,chi = param.r, param.chi

mu, nu = param.mu, param.nu

sigma_0, q = param.sigma_0, param.q

Nx , Ny = param.Nx, param.Ny

X, Y = param.X, param.Y

KX, KY = param.KX, param.KY

K2, K4, K6, K8 = param.K2, param.K4, param.K6, param.K8

k1,k2,k3,k4,k5,k6,q1,q2,q3 = param.k1, param.k2, param.k3, param.k4, param.k5, param.k6, param.q1, param.q2, param.q3

    
if param.aire_run:
    index = int(os.environ.get('SLURM_ARRAY_TASK_ID',1)) # job id number for indexing
        
        
t = param.t  # value of t to begin timestepping from

h = param.h

###############################################################################
            
            # Only saving contributions inside the wavenumber filtering range 
               
K = np.sqrt(K2)

K_number = 0

for i in range(Nx):
    for j in range(Ny):
        if K[i][j] < param.k_filtering_radius: 
            K_number = K_number+1                                         # Determine required length of list 

K_index_list = np.zeros((K_number,2),dtype = int)
K_number = 0
for i in range(Nx):
    for j in range(Ny):
        if K[i][j] < param.k_filtering_radius:
            K_index_list[K_number] = [i,j]
            K_number = K_number+1                                         # For saving data 

np.savetxt(r'dataoutput/Kindexlist.txt',K_index_list)        

###############################################################################

growthrootsannulus = fn.GrowthRootsAnnulus(mu, nu)      # zeros of growth rate function
  
###############################################################################

sigma_8 = sigma_0/(q**4) - mu*(q**2-3)/((q**2-1)**3) + nu*(1-3*q**2) / \
    (q**4*(q**2-1)**3)  # determined via symbolic calculation

sigma_6 = (2*mu-2*sigma_8*(2*q**4+2*q**2-1)-2*sigma_0) / \
    (3*q**2-1)  # determined using written analysis

sigma_4 = -(3/2*sigma_6*(q**2+1)+2*sigma_8*(q**4+q**2+1))

sigma_2 = -2*sigma_4 - 3*sigma_6 - 4*sigma_8

Lhat = sigma_0 + sigma_2 * K2 + sigma_4*K4 + sigma_6 * \
    K6 + sigma_8*K8             # FFT Linear terms

# ETD coefficients

expLhath = np.exp(Lhat*h)

expLhath2 = np.exp(Lhat*h/2)

epsilon = Lhat*h  # Ease of notation

etd1_factor = np.zeros((Nx,Ny))

etd4RK_factor1 = np.zeros((Nx, Ny))

etd4RK_factor2 = np.zeros((Nx, Ny))

etd4RK_factor3 = np.zeros((Nx, Ny))

for i in range(Nx):  # h multiplication since denominator is L^2*h not L^2*h^2
    for j in range(Ny):
        if abs(epsilon[i][j]) < 1.0e-1:             # implementing Taylor series when round off errors (caused by cancellations) significantly contribute to ETD coefficients 
            
            etd4RK_factor1[i][j] = h * (1/6 + epsilon[i][j]/6 + 3*epsilon[i][j]**2/40
                                        + epsilon[i][j]**3/45 + 5*epsilon[i][j]**4/1008)

            etd4RK_factor2[i][j] = h * (1/6 + epsilon[i][j]/12 + epsilon[i][j]**2/40
                                        + epsilon[i][j]**3/180 + epsilon[i][j]**4/1008)

            etd4RK_factor3[i][j] = h * (1/6 - epsilon[i][j]**2 /120 
                                        - epsilon[i][j]**3/360 - epsilon[i][j]**4/1680)
            
            if abs(epsilon[i][j]) <1.0e-5:                                                  # Taylor series less accurate here so implemented only small eps
                etd1_factor[i][j] = h * (1/2 + epsilon[i][j]/8 + epsilon[i][j]**2/48
                                         + epsilon[i][j]**3/384 + epsilon[i][j]**4/3840)
            else:
                etd1_factor[i][j] = h * (np.exp(epsilon[i][j]/2)-1)/epsilon[i][j]
        else:
            etd1_factor[i][j] = h * (np.exp(epsilon[i][j]/2)-1)/epsilon[i][j]
            
            etd4RK_factor1[i][j] = h * (np.exp(epsilon[i][j])*(
                4-3*epsilon[i][j]+epsilon[i][j]**2)-4-epsilon[i][j])/(epsilon[i][j]**3)

            etd4RK_factor2[i][j] = h * (2+epsilon[i][j] + np.exp(
                epsilon[i][j])*(-2+epsilon[i][j]))/(epsilon[i][j]**3)

            etd4RK_factor3[i][j] = h * (-4-3*epsilon[i][j]-epsilon[i][j]**2+np.exp(
                epsilon[i][j])*(4-epsilon[i][j]))/(epsilon[i][j]**3)

###############################################################################

U_n = np.copy(param.U_ic)       # u in physical space (current time step)

Uhat_n = fftn(param.U_ic)       # u in Fourier space

Fhat_n =fn.Fhat(U_n,Uhat_n)     # applying Fourier transform to nonlinear terms

U_nm1 = np.copy(param.U_ic)     # u at previous time step

max_U_nm1 = np.max(param.U_ic)  # max value of U_ic

max_U_t_nm1 = np.max(np.abs(ifftn(Lhat*Uhat_n +fn.Fhat(U_n,Uhat_n)).real))      # computed from PDE

criteria_1_list = np.zeros(math.ceil(param.number_of_steps/param.save_step))    # pattern classification criteria

criteria_2_list = np.copy(criteria_1_list)

criteria_3_list = np.copy(criteria_2_list)

for n_steps in range(param.number_of_steps):                # ETD4RK iteration

    # RK2 Intermediate state in Fourier Space
    Ahat_n = np.multiply(expLhath2, Uhat_n) + np.multiply(etd1_factor, Fhat_n)

    # A_n in physical space
    A_n = ifftn(Ahat_n).real

    # nonlinear terms evaluated at A_n
    FAhat_n =fn.Fhat(A_n,Ahat_n)
    
    # computing other intermediate terms for ETD4RK

    Bhat_n = np.multiply(expLhath2, Uhat_n) + np.multiply(etd1_factor, FAhat_n)

    B_n = ifftn(Bhat_n).real

    FBhat_n =fn.Fhat(B_n,Bhat_n)

    Chat_n = np.multiply(expLhath2, Ahat_n) + \
        np.multiply(etd1_factor, 2*FBhat_n-Fhat_n)

    C_n = ifftn(Chat_n).real

    FChat_n =fn.Fhat(C_n,Chat_n)             

    Uhat_np1 = (np.multiply(Uhat_n, expLhath) + np.multiply(Fhat_n, etd4RK_factor1) +
                2*np.multiply(FAhat_n + FBhat_n, etd4RK_factor2) +
                np.multiply(FChat_n, etd4RK_factor3))                           # computing u at next time step


    # get ready for the next timestep:
    Uhat_n = np.copy(Uhat_np1)
    
    
    if param.filtering:                           # implements Fourier Filtering (dealiasing) if this is chosen
        Uhat = fn.Fourier_Filtering(Uhat_n)
        Uhat_n = np.copy(Uhat)
    
    U_n = ifftn(Uhat_n).real                # computing after dealiasing, ensures U_n Uhat_n are consistent at the same timestep
    
    Fhat_n =fn.Fhat(U_n,Uhat_n)    # computing when U_n and Uhat are computed at the same timestep

    max_U_n = np.max(np.abs(U_n)) 
    
    U_t = (U_n - U_nm1)/h   # approximation of U_t, if exact wanted, use: U_t = ifftn(Lhat*Uhat_n +fn.Fhat(U_n,Uhat_n)).real 

    max_U_t = np.max(np.abs(U_t))
    
    print(t, max_U_n, max_U_t)      # printing to provide updates
    

    
    
    if n_steps %param.save_step==0: # not performing at every time step: saving data
        
        criteria_index = n_steps // param.save_step

        if n_steps ==0:
            savetype = 'w'  # write new text file (only at t=0)  
        else:
            savetype = 'a'  # append to existing text file
        
        with open(r'dataoutput/t_vals.txt',savetype) as text_file:
            np.savetxt(text_file,np.array([t]))
        
        with open(r'dataoutput/max_U_t_vals.txt',savetype) as text_file:
            np.savetxt(text_file,np.array([max_U_t]))
            
        Uhat_save = np.zeros((len(K_index_list)),dtype=complex)
        for Kindex in range(len(K_index_list)):
            [i,j] = K_index_list[Kindex]
            Uhat_save[Kindex] = Uhat_n[i][j]
           
        
        with open(r'dataoutput/Uhat_vals.txt',savetype) as text_file:
            np.savetxt(text_file,Uhat_save,fmt= param.complex_tol)

        
        criteria_1_list[criteria_index] = max_U_t/max_U_n
        
        criteria_2_list[criteria_index] = np.abs(max_U_t/max_U_n -max_U_t_nm1/max_U_nm1)/h
        
        criteria_3_list[criteria_index] = np.average(np.abs(U_t)/max_U_n)   

    if max_U_n > 100:           # breaking loop when $u \to \inf $
        break
                 
    if abs(max_U_n-max_U_nm1)/h < param.convergence_tol: #breaking loop when we have convergence: Only for equilibria as time independent
        if t>param.break_t_equilibrium:                                        # implementing break for t > some value 
            if max_U_t< param.convergence_tol:
                print(t,'converged',max_U_n)
                n_steps = n_steps+1
                break    
    
    if n_steps % param.save_U_ic ==0:                                             
        np.savetxt(r'dataoutput/new_U_ic.txt',U_n) # update initial condition in case of failed run
        
        
    # Update previous timestep values
    
    U_nm1 = np.copy(U_n)
    
    max_U_nm1 = max_U_n
    
    max_U_t_nm1 = max_U_t
    
    # Update timestepping parameters 

    t = t+h
    

np.savetxt(r'dataoutput/criteria_1_vals.txt',np.array(criteria_1_list))

np.savetxt(r'dataoutput/criteria_2_vals.txt',np.array(criteria_2_list))

np.savetxt(r'dataoutput/criteria_3_vals.txt',np.array(criteria_3_list))   

criteria_1 = criteria_1_list[-1]

criteria_2 = np.average(criteria_2_list[-param.average_cutoff:])

criteria_3 = np.average(criteria_3_list[-param.average_cutoff:])

print(f'criteria_1 = {criteria_1}, criteria_2 = {criteria_2}, criteria_3 = {criteria_3}') 

    
print('Run Finished')       # Informs user program has run successfully
    

