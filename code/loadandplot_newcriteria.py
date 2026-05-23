# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 10:31:06 2023

@author: mm18lp
"""

# VERSION 2.1.2

import numpy as np
import math
import matplotlib.pyplot as plt
import os
from skimage.filters import window

from scipy.fft import fftn, ifftn                   # was using scipy.fftpack   

import parameters as param

import functions as fn

Plots = False                # optional for producing extra plots (not required to generate data in paper)

Patch = False               # mark True if plots for patch analysis are wanted

Peaks = True                # mark True if plots of the peaks for each annulus are wanted

###############################################################################

# Importing Parameters 

r,chi = param.r, param.chi

mu, nu = param.mu, param.nu

sigma_0, q = param.sigma_0, param.q

Nx , Ny = param.Nx, param.Ny

X, Y = param.X, param.Y

KX, KY = param.KX, param.KY

K, K2, K4, K6, K8 = param.K, param.K2, param.K4, param.K6, param.K8

k1,k2,k3,k4,k5,k6,q1,q2,q3 = param.k1, param.k2, param.k3, param.k4, param.k5, param.k6, param.q1, param.q2, param.q3
    
t = param.t  # value of t to begin timestepping from

h = param.h

growthrootsannulus = fn.GrowthRootsAnnulus(mu, nu)


if param.aire_run:
    
    index = int(os.environ.get('SLURM_ARRAY_TASK_ID',1)) # job id number for indexing
        
    plt.switch_backend('agg')  # stops plots displaying
    
      
###############################################################################

# Load in saved data for plotting

K_index_list = np.loadtxt(r'dataoutput/Kindexlist.txt').astype(np.int64)

t_val = np.loadtxt(r'dataoutput/t_vals.txt')

max_U_t_list = np.loadtxt(r'dataoutput/max_U_t_vals.txt') 

Uhat_1D = np.loadtxt(r'dataoutput/Uhat_vals.txt',dtype = complex)

Uhat_2D = np.reshape(Uhat_1D,(len(t_val),len(K_index_list)))

timesteps = np.shape(Uhat_2D)[0]                                                # number of times we saved data
keptUvals = np.shape(Uhat_2D)[1]                                                # scnumber of modes we saved data for (only K< set bound)
Uhat_3D = np.zeros((timesteps,Nx,Ny),dtype=complex)                             # Uhat at each timestep

U_n_animation = np.zeros((timesteps,Nx,Ny))
for timeindex in range(timesteps):
    for Kindex in range(keptUvals):
        [i,j] = K_index_list[Kindex]
        Uhat_3D[timeindex][i][j] = Uhat_2D[timeindex][Kindex]                   # 3D array in time order 
        
    U_n_animation[timeindex] = ifftn(Uhat_3D[timeindex]).real                          # Real U_n at each timestep     


reshaped_Uhat = np.transpose(Uhat_3D,[1,2,0])                                     # simultaneously swap axes so each element in 2x2 matrix 
                                                                               # is a list of wavenumber values at different timesteps
logU = np.abs(reshaped_Uhat/(Nx*Ny))                                           # useful for plotting
                                                                               
###############################################################################

U_n = U_n_animation[-1]  

Uhat_n = Uhat_3D[-1]                                                       # Final solution U_n is last entry of U_n_animation

max_U_t = max_U_t_list[-1]

max_U_n = np.max(abs(U_n))

colourspectrum = np.linspace(np.min(U_n),np.max(U_n),25)    # for colourbar

###############################################################################

                        # Optional Plots: patterned state, fourier spectrum, time series of time derivative, time series of individual mode amplitudes.
            
if Plots:
    figs, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(16,12), num = f'chi = {chi}')
    ax1.set_xlabel('$x/{2\pi}$')
    ax1.set_ylabel('$y/{2\pi}$')
    ax1.set_aspect('equal')
    pcm = ax1.contourf(X/(2*math.pi), Y/(2*math.pi), U_n, levels=colourspectrum,cmap = 'inferno')
    figs.colorbar(pcm,ax=ax1)
    figs.suptitle(f'r={round(r,3)} $\chi$={chi} $\mu$={round(mu,3)} $\\nu$={round(nu,3)}', fontsize=16)  # patterned solution u
    
    ax2.set_xlabel('$k_x$')
    ax2.set_ylabel('$k_y$')
    ax2.set_aspect('equal','box')
    circle1 = plt.Circle((0, 0), q, edgecolor='r',fill=None,linewidth=1.5)
    circle2 = plt.Circle((0,0),1,edgecolor='b',fill=None,linewidth=1.5)
    ax2.add_patch(circle1)
    ax2.add_patch(circle2)
    
    ax3.set_xlabel('$t$')
    ax3.set_ylabel('max$(u_t)$')
    ax3.semilogy(t_val,max_U_t_list,'k',markersize=1)                       # time series for largest element in u_t
    
    ax4.set_xlabel('$t$')
    ax4.set_ylabel('$\log{\hat{u}}$')
    
    Uhatmax = np.max(abs(Uhat_n)/(Nx*Ny))
    
    scaledUhat = np.log10(abs(Uhat_n/(Nx*Ny))/Uhatmax)
    
    for i in range(Nx):
        i = Nx-1-i  
        for j in range(Ny):
            j = Ny-1-j
            u = scaledUhat[i][j]
            if u <-4:
                continue
            else:
                if u> -0.5:                                                     # Fourier Spectrum
                    ax2.plot(KX[i][j],KY[i][j],'k.',markersize=5)                
                elif u>-1:
                    ax2.plot(KX[i][j],KY[i][j],color = '0.3', marker = '.', markersize=4)
                elif u> -2:            
                    ax2.plot(KX[i][j],KY[i][j],color = '0.55', marker = '.',markersize=3)
                elif u>-3:
                    ax2.plot(KX[i][j],KY[i][j],color = '0.75', marker = '.',markersize=3)
                else:
                    ax2.plot(KX[i][j],KY[i][j],color = '0.9', marker = '.',markersize=3)
                    
            k = K[i][j]
            if k < 3.1:                                             # colour coding mode amplitudes based off wavenumber
                if k > 2*growthrootsannulus[3]: 
                    ax4.semilogy(t_val,logU[i][j],'y',linewidth=0.5)
                    
                elif k > growthrootsannulus[3]:
                    ax4.semilogy(t_val,logU[i][j],'c',linewidth=0.5)
    
                elif k > growthrootsannulus[2]:
                    ax4.semilogy(t_val,logU[i][j],'m',linewidth=0.5)
    
                elif k > growthrootsannulus[1]:
                    ax4.semilogy(t_val,logU[i][j],'g',linewidth=0.5)
    
                elif k > growthrootsannulus[0]:
                    ax4.semilogy(t_val,logU[i][j],'r',linewidth=0.5)
    
                elif k > 1e-8:
                    ax4.semilogy(t_val,logU[i][j],'b',linewidth=0.5)
    
                else:
                    ax4.semilogy(t_val,logU[i][j],'k',linewidth=0.5)
    plt.tight_layout()
    if param.aire_run:
        figs.savefig(f'../../Subplots/Subplots_run{index}.png',format='png')
        

###############################################################################

interval_width = param.annulus_interval_width   # size of annulus

Uhat = np.abs(Uhat_n)/(Nx*Ny)

Nq=0
N1 = 0

for i in range(Nx):                                                             # Determing number of entries in annulus
    for j in range(Ny):
        if q-interval_width/1.5< K[i][j]< q+ interval_width/1.5:
            Nq=Nq+1

        
        elif 1-interval_width/1.5 < K[i][j] < 1 + interval_width/1.5 :
            N1=N1+1
            
k_q_annulus = np.zeros(Nq,dtype = complex)      # create arrays of correct size

u_k_q = np.zeros(Nq, dtype = complex)

k_1_annulus = np.zeros(N1,dtype = complex)

u_k_1 = np.zeros(N1,dtype = complex)                                              

Nq=0
N1=0

# creating arrays for value of amplitudes for each mode in the annuli

for i in range(Nx):
    for j in range(Ny):
        if q-interval_width/1.5< K[i][j]< q+ interval_width/1.5:
                    k_q_annulus[Nq] = KX[i][j]+1j*KY[i][j]
                    u_k_q[Nq] = Uhat[i][j]
                    Nq = Nq+1

        
        elif 1-interval_width/1.5 < K[i][j] < 1 + interval_width/1.5 :
            k_1_annulus[N1] = KX[i][j]+1j*KY[i][j]
            u_k_1[N1] = Uhat[i][j]
            N1 = N1+1



k_1_phases = np.mod(np.angle(k_1_annulus,deg=True),360)     # phases/angles of the modes in the annuli
    
k_q_phases = np.mod(np.angle(k_q_annulus,deg=True),360)


bin_angle_reference = np.arange(0,360,param.bin_width)              # list of angles to bin

bin_number = len(bin_angle_reference)                   # number of bins/segments

u_k_1_bin_index = np.zeros(len(k_1_annulus))

u_k_q_bin_index = np.zeros(len(k_q_annulus))

# assigning each mode to a bin based off its phase
for indices in range(N1):
    k = k_1_annulus[indices]
    u = u_k_1[indices]
    angle = k_1_phases[indices]
    bin_index = int(angle//param.bin_width) 
    
    u_k_1_bin_index[indices] = bin_index
    
    
for indices in range(Nq):
    k = k_q_annulus[indices]
    u = u_k_q[indices]
    angle = k_q_phases[indices]
    bin_index = int(angle//param.bin_width)       # integer bin index
    
    u_k_q_bin_index[indices] = bin_index
  

# creating lists for average value of mode amplitude in each bin
    
k_1_average_list = np.zeros(bin_number)
k_q_average_list = np.zeros(bin_number)

k_1_max_list = np.zeros(bin_number)
k_q_max_list = np.zeros(bin_number)

# computing average and maximum of mode amplitudes in each bin
for bin in range(bin_number):
   k_1_average =  np.average(np.abs(u_k_1), weights = (u_k_1_bin_index == bin)) # averaging over elements with each bin number
   k_q_average =  np.average(np.abs(u_k_q), weights = (u_k_q_bin_index == bin))
   
   k_1_max = np.max(np.abs(u_k_1), where= (u_k_1_bin_index==bin),initial=0)     # calculating max in each bin (computing over indices with correct bin number)
   k_q_max = np.max(np.abs(u_k_q), where= (u_k_q_bin_index==bin),initial=0)     # where replaces any index not satisfying condition to zero, then computes max over rest
   
   k_1_average_list[bin] = k_1_average                                          # saving elements to lists
   k_q_average_list[bin] = k_q_average
   
   k_1_max_list[bin] = k_1_max
   k_q_max_list[bin] = k_q_max

###############################################################################

# Pattern Classification criteria: number of peaks and fuzziness    
k_1_fuzzy = False
k_q_fuzzy = False

k1_peaks = 0
kq_peaks = 0 
                                                                 # True/False and using max value in bin and changing peak criteria
                                                                 
k_1_compare = max(k_1_max_list)/3           # cutoff values for classification of Peak for each critical circle
k_q_compare = max(k_q_max_list)/3

k_max_compare = max(k_1_compare,k_q_compare)        # determining largest cutoff value across both annuli

if k_max_compare == k_1_compare:        # if largest  peak is in k=1 annulus
    
    if max(k_q_max_list) > k_1_compare:         # largest contribution from k=q is large enough to be classified as a Peak 

        for i in range(-2,bin_number-2):        # iterating over all bins
            if k_1_compare >1e-8:           
                u_k1 = k_1_max_list[i]                                          # iterating over all k=1 bins
                if  (u_k1 > k_1_max_list[i+1] and u_k1 > k_1_max_list[i-1]  
                     and u_k1 > k_1_max_list[i-2] and u_k1 > k_1_max_list[i+2]) :  # bigger than the two averages either side     
                    if u_k1>k_max_compare:                                          # check if large enough to be classed as a Peak
                        
                        k1_peaks +=1
                        
                        print('k=1', bin_angle_reference[i])
                    elif u_k1 > 5e-3:                                           # not large enough to be a peak, but large enough to contribute to 'fuzziness'
                        k_1_fuzzy = True
                elif u_k1 > 5e-3:
                    k_1_fuzzy = True
                
            if k_q_compare >1e-8:                                               
                u_kq = k_q_max_list[i]                                           # iterating over all k=q bins (repeat above)
                if u_kq>k_q_compare:
                    if  (u_kq > k_q_max_list[i+1] and u_kq > k_q_max_list[i-1] 
                     and u_kq > k_q_max_list[i-2] and u_kq > k_q_max_list[i+2]) : 

                        kq_peaks +=1
                        
                        print('k=q',bin_angle_reference[i])
                    elif u_kq > 5e-3:
                        k_q_fuzzy = True 
                elif u_kq > 5e-3:
                    k_q_fuzzy = True
                    
    else:                                                                       # if all k=q contributions are too small to be a Peak
        for i in range(-2,bin_number-2):
            if k_1_compare >1e-8:                                               # k=1 analysis remains unchanged
                u_k1 = k_1_max_list[i] 
                if  (u_k1 > k_1_max_list[i+1] and u_k1 > k_1_max_list[i-1] 
                     and u_k1 > k_1_max_list[i-2] and u_k1 > k_1_max_list[i+2]) :  # bigger than the two averages either side     
                    if u_k1>k_max_compare:
                        
                        k1_peaks +=1
                        
                        print('k=1', bin_angle_reference[i])
                    elif u_k1 > 5e-3:
                        k_1_fuzzy = True
                elif u_k1 > 5e-3:
                    k_1_fuzzy = True
                
            if k_q_compare >1e-8:                                               # check only for fuzziness for k=q
                u_kq = k_q_max_list[i]
                if u_kq > 5e-3:
                    k_q_fuzzy = True 
                    
elif k_max_compare == k_q_compare:                                              # if largest contribution is in k=q annulus
    
    if max(k_1_max_list) > k_q_compare:                                         # largest contribution from k=1 is large enough to be classified as a Peak 

        for i in range(-2,bin_number-2):                                        # repeating above analysis
            if k_1_compare >1e-8:
                u_k1 = k_1_max_list[i] 
                if u_k1>k_1_compare:
                    if  (u_k1 > k_1_max_list[i+1] and u_k1 > k_1_max_list[i-1] 
                         and u_k1 > k_1_max_list[i-2] and u_k1 > k_1_max_list[i+2]) :  # bigger than the two averages either side     

                        
                        k1_peaks +=1
                        
                        print('k=1', bin_angle_reference[i])
                    elif u_k1 > 5e-3:
                        k_1_fuzzy = True
                elif u_k1 > 5e-3:
                    k_1_fuzzy = True
                
            if k_q_compare >1e-8:
                u_kq = k_q_max_list[i]
                if u_kq>k_max_compare:
                    if  (u_kq > k_q_max_list[i+1] and u_kq > k_q_max_list[i-1] 
                     and u_kq > k_q_max_list[i-2] and u_kq > k_q_max_list[i+2]) : 

                        kq_peaks +=1
                        
                        print('k=q',bin_angle_reference[i])
                    elif u_kq > 5e-3:
                        k_q_fuzzy = True 
                            
                elif u_kq > 5e-3:
                    k_q_fuzzy = True
                    
    else:
        for i in range(-2,bin_number-2):                                        # check only for fuzziness in k=1 annulus
            if k_1_compare >1e-8:
                u_k1 = k_1_max_list[i] 
                if u_k1 > 5e-3:
                    k_1_fuzzy = True
                
            if k_q_compare >1e-8:
                u_kq = k_q_max_list[i]
                if u_kq>k_max_compare:
                    if  (u_kq > k_q_max_list[i+1] and u_kq > k_q_max_list[i-1] 
                     and u_kq > k_q_max_list[i-2] and u_kq > k_q_max_list[i+2]) : 

                        kq_peaks +=1
                        
                        print('k=q',bin_angle_reference[i])
                    elif u_kq > 5e-3:
                        k_q_fuzzy = True 
                elif u_kq > 5e-3:
                    k_q_fuzzy = True                        

# Using the criteria \mathcal{C}_1, \mathcal{C}_2 and \mathcal{C}_3 to classify time-dependence
Equilibrium = False

Slow = False

Fast = False

Small_Vars = False

Small_Spatial_Change = False

Large_Spatial_Change = False


# Load in criteria from ETD4RK iterations

criteria_1_list = np.loadtxt(r'dataoutput/criteria_1_vals.txt')

criteria_2_list = np.loadtxt(r'dataoutput/criteria_2_vals.txt')

criteria_3_list = np.loadtxt(r'dataoutput/criteria_3_vals.txt')


criteria_1 = criteria_1_list[-1]

criteria_2 = np.average(criteria_2_list[-param.average_cutoff:])        # discarding the first param.average_cutoff % of values: reduce impact of transients

criteria_3 = np.average(criteria_3_list[-param.average_cutoff:])
    

if criteria_1 < 5e-8:
    Equilibrium = True            # equilibirum
    
elif criteria_1 < 1e-3:
    Slow = True           # slow evolving dynamics (small) 
    if criteria_2 < 1e-4:
        Small_Vars = True
    if criteria_3 > 5e-3:
        Large_Spatial_Change = True
    elif criteria_3 > 5e-4:
        Small_Spatial_Change = True
else:
    Fast = True
    if criteria_2 < 1e-4:
        Small_Vars = True
    if criteria_3 > 5e-3:
        Large_Spatial_Change = True
    elif criteria_3 > 5e-4:
        Small_Spatial_Change = True
    
        
print(f'criteria_1 = {criteria_1}, criteria_2 = {criteria_2}, criteria_3 = {criteria_3}')
print(f'Equilibrium = {Equilibrium}, Slow = {Slow}, Fast = {Fast}, Small_Vars = {Small_Vars},'  
      f' Small_Spatial_Change = {Small_Spatial_Change}, Large_Spatial_Change = {Large_Spatial_Change}')

#saving pattern classification criteria      
code = [k1_peaks,kq_peaks,k_1_fuzzy,k_q_fuzzy,Equilibrium, Slow, Fast, Small_Vars, Small_Spatial_Change,Large_Spatial_Change]

if param.aire_run:
    with open('../../pythonfiles/patterncodenew.txt','a') as text_file:
        np.savetxt(text_file,[[r,chi,k1_peaks,kq_peaks,k_1_fuzzy,k_q_fuzzy,Equilibrium, Slow, Fast, Small_Vars, Small_Spatial_Change,Large_Spatial_Change]])

if param.aire_run:
    with open('../../pythonfiles/criteria_values.txt','a') as text_file:   # saving values of criteria
        np.savetxt(text_file,[[r,chi,criteria_1,criteria_2,criteria_3]])

if Plots:               # if wanting to plot diagrams for Peaks
    if  Peaks:
        bigfigure = plt.figure('U_Peaks',figsize=(13,10))
        subfigs = bigfigure.subfigures(1,2)
        subfigs[0].suptitle(f'r={round(r,3)} $\chi$={chi} $\mu$={round(mu,3)} $\\nu$={round(nu,3)}', fontsize=12)
        subfigs[1].suptitle(f'$k_1$ = {k1_peaks}  $k_q$ = {kq_peaks}  $k_{{1f}}$ = {k_1_fuzzy}  $k_{{qf}}$ = {k_q_fuzzy} \n  E = {Equilibrium}  S = {Slow}  F = {Fast}  SV = {Small_Vars}  SSC = {Small_Spatial_Change}  LSC = {Large_Spatial_Change}', fontsize=12)
        
        
        colourspectrum = np.linspace(np.min(U_n),np.max(U_n),25)
        
        
        axsLeft =  subfigs[0].subplots(2,1)
        for nn, ax in enumerate(axsLeft):
            if nn==0:
                ax.set_xlabel('$x/{2\pi}$')
                ax.set_ylabel('$y/{2\pi}$')
                axes = plt.gca()
                axes.xaxis.label.set_size(20)
                axes.yaxis.label.set_size(20)
                axes.set_aspect('equal')
                pcm = ax.contourf(X/(2*math.pi), Y/(2*math.pi), U_n, levels=colourspectrum,cmap = 'inferno') # plot of pattern u 
        
            elif nn==1:
        
                ax.set_xlabel('$k_x$')
                ax.set_ylabel('$k_y$')
                ax.set_xlim([-1.5,1.5])
                ax.set_ylim([-1.5,1.5])
                axes2 = plt.gca()
                axes2.xaxis.label.set_size(20)
                axes2.yaxis.label.set_size(20)
                axes2.set_aspect('equal','box')
                circle1 = plt.Circle((0, 0), q, edgecolor='r',fill=None,linewidth=1.5)
                circle2 = plt.Circle((0,0),1,edgecolor='b',fill=None,linewidth=1.5)
                plt.gca().add_patch(circle1)
                plt.gca().add_patch(circle2)
                plt.tight_layout(rect=[None,None,None,0.95])   # needed to add space between subtitle and plots
        
                Uhatmax = np.max(abs(Uhat_n)/(Nx*Ny))
        
                scaledUhat = np.log10(abs(Uhat_n/(Nx*Ny))/Uhatmax)
        
                for i in range(Nx):
                    for j in range(Ny):
                        u = scaledUhat[i][j]
                        if u <-4:
                            continue
                        else:
                            if u> -0.5:
                                ax.plot(KX[i][j],KY[i][j],'k.',markersize=5)
                            elif u>-1:
                                ax.plot(KX[i][j],KY[i][j],color = '0.3', marker = '.', markersize=4)
                            elif u> -2:            
                                ax.plot(KX[i][j],KY[i][j],color = '0.55', marker = '.',markersize=3)
                            elif u>-3:
                                ax.plot(KX[i][j],KY[i][j],color = '0.75', marker = '.',markersize=3)
                            else:
                                ax.plot(KX[i][j],KY[i][j],color = '0.9', marker = '.',markersize=3)
                ax.contour(KX,KY,K,[q-interval_width/1.5, q+interval_width, 1- interval_width/1.5, 1+interval_width/1.5])
                pos = ax.get_position()
                newpos = [pos.x0-0.08,pos.y0,pos.width,pos.height]
                ax.set_position(newpos)
        subfigs[0].colorbar(pcm,ax=axsLeft[0],location='right')                 # plotting Fourier spectrum
        
        axsRight = subfigs[1].subplots(2,1,sharex=True, sharey=True)            # Plotting the max and average of each bin for both annuli.
        for nn, ax in enumerate(axsRight):
            if nn==0:
                ax.set_ylabel('Amplitude $k=1$')
                ax.bar(bin_angle_reference,np.array(k_1_max_list),width=2,color = 'c',log=False,label='max')
                ax.bar(bin_angle_reference,np.array(k_1_average_list),width=2,color='b',log=False,label='average')
                ax.legend(loc='upper left')
            elif nn==1:       
                ax.set_xlabel('$k_{\\theta}$')
                ax.set_ylabel('Amplitude $k=q$')
                ax.bar(bin_angle_reference,np.array(k_q_max_list),width=2,color='c',log=False,label='max')
                ax.bar(bin_angle_reference,np.array(k_q_average_list),width=2,color='b',log=False,label='average')
                ax.legend(loc='upper left')
        if param.aire_run:
            bigfigure.savefig(f'../../Peaks_New/Peaks_run{index}.png',format='png')
    

###############################################################################
                    # Peaks_Patch plots - large box only
                    
# Patch analysis: used to classify when defects are present in pattern

if param.large_box:  # only computing patches for large box case
    
# constructing small box for patch analysis
    Lx_small = param.aspect_ratio_x * 2 * math.pi * param.number_of_wavelengths_x_small
    
    Ly_small = Lx_small
    
    Nx_small = int(Nx*param.number_of_wavelengths_x_small/param.number_of_wavelengths_x)
    
    Ny_small = Nx_small
    
    x_small = np.arange(0.0, Nx_small)*(Lx_small/(Nx_small))
    
    y_small = x_small
    
    X_small_1, Y_small_1 = np.meshgrid(x_small, y_small, indexing='ij')
    
    Nx_small_shape = len(x_small)
    
    Ny_small_shape = len(y_small)
    
    kx_small = np.arange(0, Nx_small) * 1.0
    for i in range(int(Nx_small_shape/2+1), Nx_small_shape):
        kx_small[i] = i-Nx_small
    
    # y component of wavevector
    ky_small = np.arange(0, Ny_small) * 1.0
    for i in range(int(Ny_small_shape/2+1), Ny_small_shape):
        ky_small[i] = i-Ny_small
    
    
    # Scale integers to physical wavenumbers
    
    kx_small = kx_small*2*math.pi/Lx_small
    
    # Squared wavevectors
    kx2_small = np.square(kx_small)
    
    ky_small = ky_small*2*math.pi/Ly_small
    
    ky2_small = np.square(ky_small)
    
    # Ensures correct dimensions on wavevectors
    KX_small, KY_small = np.meshgrid(kx_small, ky_small, indexing='ij')
    
    KX2_small, KY2_small = np.meshgrid(kx2_small, ky2_small, indexing='ij')
    
    
    K2_small = KX2_small + KY2_small
    
    K_small = np.sqrt(K2_small)

###############################################################################  

# Performing patch analysis on 3 regions of the domain: to try find a region which does not contain defects

    U_patch_1 = U_n[0:Nx_small_shape,0:Ny_small_shape]
    
    U_patch_2 = U_n[-1-Nx_small_shape:-1,(Ny-Ny_small_shape)//2:(Ny+Ny_small_shape)//2]
    
    U_patch_3 = U_n[(Nx-Nx_small_shape)//2:(Nx+Nx_small_shape)//2,-1-Ny_small_shape:-1]
    
    X_small_2 = X_small_1 + np.ones(X_small_1.shape)*X[-1-Nx_small_shape,0]
    
    Y_small_2 = Y_small_1 + np.ones(Y_small_1.shape)*Y[0,(Ny-Ny_small_shape)//2]
    
    X_small_3 = X_small_1 + np.ones(X_small_1.shape)*X[(Nx-Nx_small_shape)//2,0]
    
    Y_small_3 = Y_small_1 + np.ones(Y_small_1.shape)*Y[0,-1-Ny_small_shape]
    
    
    U_patch_list = [U_patch_1, U_patch_2, U_patch_3]
    
    X_small_list = [X_small_1,X_small_2,X_small_3]
    
    Y_small_list = [Y_small_1,Y_small_2,Y_small_3]

    
    for patchindex in range(len(U_patch_list)):
        U_patch = U_patch_list[patchindex]    
        X_small = X_small_list[patchindex]
        Y_small = Y_small_list[patchindex]
    
        wimage = U_patch * window('hann', U_patch.shape)                # multiplied by Hann window to reduce large frequency contributions
    
        
        wUhat = fftn(wimage)
        abswUhat = np.abs(wUhat)/(Nx_small_shape*Ny_small_shape)
          
        # Repeating peak analysis for small box (patch analysis)
        Nq=0
        N1 = 0
        
        interval_width = 0.1    # annulus width
    
        for i in range(Nx_small_shape):                                                             # Determing number of entries in annulus
            for j in range(Ny_small_shape):
                if q-interval_width/1.5< K[i][j]< q+ interval_width/1.5:
                    Nq=Nq+1
    
                
                elif 1-interval_width/1.5 < K[i][j] < 1 + interval_width/1.5 :
                    N1=N1+1
                    
        k_q_annulus = np.zeros(Nq,dtype = complex)
    
        u_k_q = np.zeros(Nq, dtype = complex)                                        
    
    
        k_1_annulus = np.zeros(N1,dtype = complex)
    
        u_k_1 = np.zeros(N1,dtype = complex)                                             
    
        Nq=0
        N1=0
    
    
        for i in range(Nx_small_shape):
            for j in range(Ny_small_shape):
                if q-interval_width/1.5< K_small[i][j]< q+ interval_width/1.5:
                            k_q_annulus[Nq] = KX_small[i][j]+1j*KY_small[i][j]
                            u_k_q[Nq] = abswUhat[i][j]
                            Nq = Nq+1
    
                
                elif 1-interval_width/1.5 < K_small[i][j] < 1 + interval_width/1.5 :
                    k_1_annulus[N1] = KX_small[i][j]+1j*KY_small[i][j]
                    u_k_1[N1] = abswUhat[i][j]
                    N1 = N1+1
    
    
    
        k_1_phases = np.mod(np.angle(k_1_annulus,deg=True),360)
            
        k_q_phases = np.mod(np.angle(k_q_annulus,deg=True),360)
    
        bin_width = 2*param.bin_width  # double the width of normal bin as less modes
    
        bin_angle_reference = np.arange(0,360,bin_width)              # angles to bin
    
        bin_number = len(bin_angle_reference)
    
        u_k_1_bin_index = np.zeros(len(k_1_annulus))
        
        u_k_q_bin_index = np.zeros(len(k_q_annulus))
        
        for indices in range(N1):
            k = k_1_annulus[indices]
            u = u_k_1[indices]
            angle = k_1_phases[indices]
            bin_index = int(angle//bin_width) 
            
            u_k_1_bin_index[indices] = bin_index
            
            
        for indices in range(Nq):
            k = k_q_annulus[indices]
            u = u_k_q[indices]
            angle = k_q_phases[indices]
            bin_index = int(angle//bin_width)       # integer bin index
            
            u_k_q_bin_index[indices] = bin_index
          
            
        k_1_average_list = np.zeros(bin_number)
        k_q_average_list = np.zeros(bin_number)
        
        k_1_max_list = np.zeros(bin_number)
        k_q_max_list = np.zeros(bin_number)
        
        
        for bin in range(bin_number):
            k_1_average =  np.average(np.abs(u_k_1), weights = (u_k_1_bin_index == bin)) # averaging over elements with each bin number
            k_q_average =  np.average(np.abs(u_k_q), weights = (u_k_q_bin_index == bin))
           
            k_1_max = np.max(np.abs(u_k_1), where= (u_k_1_bin_index==bin),initial=0)     # calculating max in each bin (computing over indices with correct bin number)
            k_q_max = np.max(np.abs(u_k_q), where= (u_k_q_bin_index==bin),initial=0)     # where replaces any index not satisfying condition to zero, then computes max over rest
           
            k_1_average_list[bin] = k_1_average
            k_q_average_list[bin] = k_q_average
           
            k_1_max_list[bin] = k_1_max
            k_q_max_list[bin] = k_q_max
    
        
        
        k_1_fuzzy = False
        k_q_fuzzy = False
        
        k1_peaks = 0
        kq_peaks = 0 
            
                                                                  # True/False and using max value in bin and changing peak criteria
        k_1_compare = max(k_1_max_list)/3
        k_q_compare = max(k_q_max_list)/3
        
        k_max_compare = max(k_1_compare,k_q_compare)
        
        if k_max_compare == k_1_compare:
            
            if max(k_q_max_list) > k_1_compare: 
        
                for i in range(-2,bin_number-2):
                    if k_1_compare >1e-8:
                        u_k1 = k_1_max_list[i] 
                        if  (u_k1 > k_1_max_list[i+1] and u_k1 > k_1_max_list[i-1] 
                              and u_k1 > k_1_max_list[i-2] and u_k1 > k_1_max_list[i+2]) :  # bigger than the two averages either side     
                            if u_k1>k_max_compare:
                                
                                k1_peaks +=1
                                
                                print('k=1', bin_angle_reference[i])
                            elif u_k1 > 5e-3:
                                k_1_fuzzy = True
                        elif u_k1 > 5e-3:
                            k_1_fuzzy = True
                        
                    if k_q_compare >1e-8:
                        u_kq = k_q_max_list[i]
                        if u_kq>k_q_compare:
                            if  (u_kq > k_q_max_list[i+1] and u_kq > k_q_max_list[i-1] 
                              and u_kq > k_q_max_list[i-2] and u_kq > k_q_max_list[i+2]) : 
    
                                kq_peaks +=1
                                
                                print('k=q',bin_angle_reference[i])
                            elif u_kq > 5e-3:
                                k_q_fuzzy = True 
                        elif u_kq > 5e-3:
                            k_q_fuzzy = True
                            
            else:
                for i in range(-2,bin_number-2):
                    if k_1_compare >1e-8:
                        u_k1 = k_1_max_list[i] 
                        if  (u_k1 > k_1_max_list[i+1] and u_k1 > k_1_max_list[i-1] 
                              and u_k1 > k_1_max_list[i-2] and u_k1 > k_1_max_list[i+2]) :  # bigger than the two averages either side     
                            if u_k1>k_max_compare:
                                
                                k1_peaks +=1
                                
                                print('k=1', bin_angle_reference[i])
                            elif u_k1 > 5e-3:
                                k_1_fuzzy = True
                        elif u_k1 > 5e-3:
                            k_1_fuzzy = True
                        
                    if k_q_compare >1e-8:
                        u_kq = k_q_max_list[i]
                        if u_kq > 5e-3:
                            k_q_fuzzy = True 
                            
        elif k_max_compare == k_q_compare:
            
            if max(k_1_max_list) > k_q_compare: 
        
                for i in range(-2,bin_number-2):
                    if k_1_compare >1e-8:
                        u_k1 = k_1_max_list[i] 
                        if u_k1>k_1_compare:
                            if  (u_k1 > k_1_max_list[i+1] and u_k1 > k_1_max_list[i-1] 
                                  and u_k1 > k_1_max_list[i-2] and u_k1 > k_1_max_list[i+2]) :  # bigger than the two averages either side     
    
                                
                                k1_peaks +=1
                                
                                print('k=1', bin_angle_reference[i])
                            elif u_k1 > 5e-3:
                                k_1_fuzzy = True
                        elif u_k1 > 5e-3:
                            k_1_fuzzy = True
                        
                    if k_q_compare >1e-8:
                        u_kq = k_q_max_list[i]
                        if u_kq>k_max_compare:
                            if  (u_kq > k_q_max_list[i+1] and u_kq > k_q_max_list[i-1] 
                              and u_kq > k_q_max_list[i-2] and u_kq > k_q_max_list[i+2]) : 
    
                                kq_peaks +=1
                                
                                print('k=q',bin_angle_reference[i])
                            elif u_kq > 5e-3:
                                k_q_fuzzy = True 
                                    
                        elif u_kq > 5e-3:
                            k_q_fuzzy = True
                            
            else:
                for i in range(-2,bin_number-2):
                    if k_1_compare >1e-8:
                        u_k1 = k_1_max_list[i] 
                        if u_k1 > 5e-3:
                            k_1_fuzzy = True
                        
                    if k_q_compare >1e-8:
                        u_kq = k_q_max_list[i]
                        if u_kq>k_max_compare:
                            if  (u_kq > k_q_max_list[i+1] and u_kq > k_q_max_list[i-1] 
                              and u_kq > k_q_max_list[i-2] and u_kq > k_q_max_list[i+2]) : 
    
                                kq_peaks +=1
                                
                                print('k=q',bin_angle_reference[i])
                            elif u_kq > 5e-3:
                                k_q_fuzzy = True 
                        elif u_kq > 5e-3:
                            k_q_fuzzy = True                        
    
        # saving patch analysis classification
        
        if param.aire_run:
            with open('../../pythonfiles/patterncodepatch.txt','a') as text_file:
                np.savetxt(text_file,[[r,chi,patchindex+1,k1_peaks,kq_peaks,k_1_fuzzy,k_q_fuzzy]])
          
        if Plots:
            
            if Patch:   # change to Patch = True if want these plotted
            
                bigfigure = plt.figure(f'U_patch_{patchindex+1} Peaks',figsize=(13,10))
                subfigs = bigfigure.subfigures(1,2)
            
                colourspectrum = np.linspace(np.min(U_n),np.max(U_n),25)
                
            
                axsLeft =  subfigs[0].subplots(2,1)
                for nn, ax in enumerate(axsLeft):
                    if nn==0:
                        ax.set_xlabel('$x/{2\pi}$')
                        ax.set_ylabel('$y/{2\pi}$')
                        axes = plt.gca()
                        axes.xaxis.label.set_size(20)
                        axes.yaxis.label.set_size(20)
                        axes.set_aspect('equal')
                        pcm = ax.contourf(X_small/(2*math.pi), Y_small/(2*math.pi), wimage, levels=colourspectrum,cmap = 'inferno')
                
                    elif nn==1:
                
                        ax.set_xlabel('$k_x$')
                        ax.set_ylabel('$k_y$')
                        ax.set_xlim([-1.5,1.5])
                        ax.set_ylim([-1.5,1.5])
                        axes2 = plt.gca()
                        axes2.xaxis.label.set_size(20)
                        axes2.yaxis.label.set_size(20)
                        axes2.set_aspect('equal','box')
                        circle1 = plt.Circle((0, 0), q, edgecolor='r',fill=None,linewidth=1.5)
                        circle2 = plt.Circle((0,0),1,edgecolor='b',fill=None,linewidth=1.5)
                        plt.gca().add_patch(circle1)
                        plt.gca().add_patch(circle2)
                        plt.tight_layout(rect=[None,None,None,0.95])   # needed to add space between subtitle and plots
                
                        wUhat = fftn(wimage)
                
                        wUhatmax = np.max(abs(wUhat)/(Nx_small_shape*Ny_small_shape))
                
                        wscaledUhat = np.log10(abs(wUhat/(Nx_small_shape*Ny_small_shape))/wUhatmax)
                        
                        for i in range(Nx_small_shape):
                            for j in range(Ny_small_shape):
                                u = wscaledUhat[i][j]
                                if u <-4:
                                    continue
                                else:
                                    if u> -0.5:
                                        ax.plot(KX_small[i][j],KY_small[i][j],'k.',markersize=5)
                                    elif u>-1:
                                        ax.plot(KX_small[i][j],KY_small[i][j],color = '0.3', marker = '.', markersize=4)
                                    elif u> -2:            
                                        ax.plot(KX_small[i][j],KY_small[i][j],color = '0.55', marker = '.',markersize=3)
                                    elif u>-3:
                                        ax.plot(KX_small[i][j],KY_small[i][j],color = '0.75', marker = '.',markersize=3)
                                    else:
                                        ax.plot(KX_small[i][j],KY_small[i][j],color = '0.9', marker = '.',markersize=3)
                        ax.contour(KX_small,KY_small,K_small,[q-interval_width/1.5, q+interval_width, 1- interval_width/1.5, 1+interval_width/1.5])
                        pos = ax.get_position()
                        newpos = [pos.x0-0.08,pos.y0,pos.width,pos.height]
                        ax.set_position(newpos)
                subfigs[0].colorbar(pcm,ax=axsLeft[0],location='right')
                subfigs[0].suptitle(f'r={round(r,3)} $\chi$={chi} $\mu$={round(mu,3)} $\\nu$={round(nu,3)}', fontsize=12)
                
                axsRight = subfigs[1].subplots(2,1,sharex=True, sharey=True)
                for nn, ax in enumerate(axsRight):
                    if nn==0:
                        ax.set_ylabel('Amplitude $k=1$')
                        ax.bar(bin_angle_reference,np.array(k_1_max_list),width=2,color = 'c',log=False,label='max')
                        ax.bar(bin_angle_reference,np.array(k_1_average_list),width=2,color='b',log=False,label='average')
                        ax.legend(loc='upper left')
                    elif nn==1:       
                        ax.set_xlabel('$k_{\\theta}$')
                        ax.set_ylabel('Amplitude $k=q$')
                        ax.bar(bin_angle_reference,np.array(k_q_max_list),width=2,color='c',log=False,label='max')
                        ax.bar(bin_angle_reference,np.array(k_q_average_list),width=2,color='b',log=False,label='average')
                        ax.legend(loc='upper left')
                subfigs[1].suptitle(f'$k_1$ = {k1_peaks}  $k_q$ = {kq_peaks}  $k_{{1f}}$ = {k_1_fuzzy}  $k_{{qf}}$ = {k_q_fuzzy} \n  E = {Equilibrium}  S = {Slow}  F = {Fast}  SV = {Small_Vars}  SSC = {Small_Spatial_Change}  LSC = {Large_Spatial_Change}', fontsize=12)
                if param.aire_run:
                    bigfigure.savefig(f'../../Peaks_Patch/Peaks_run{index}_U_patch_{patchindex+1}.png',format='png')

print(f'criteria_1 = {criteria_1}, criteria_2 = {criteria_2}, criteria_3 = {criteria_3}')
print(f'Equilibrium = {Equilibrium}, Slow = {Slow}, Fast = {Fast}, Small_Vars = {Small_Vars},'  
      f' Small_Spatial_Change = {Small_Spatial_Change}, Large_Spatial_Change = {Large_Spatial_Change}')        


print('Plotting Complete')

