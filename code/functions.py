# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 11:59:52 2025

@author: mm18lp
"""

# VERSION 2.1 (updated Fourier Filtering)

import numpy as np

from scipy.fft import fftn, ifftn                      

import parameters as param
###############################################################################
        # Import Parameters from parameters file (for ease of notation)
                    
Q1, Q2, Q3, C1, C2, C3 = param.Q1, param.Q2, param.Q3, param.C1, param.C2, param.C3

r,chi = param.r, param.chi

mu, nu = param.mu, param.nu

sigma_0, q = param.sigma_0, param.q

Nx , Ny = param.Nx, param.Ny

KX, KY = param.KX, param.KY

K, K2, K4, K6, K8 = param.K, param.K2, param.K4, param.K6, param.K8

###############################################################################
                              # Functions

def U_squared(U):
    return np.square(U)

# U lap U in physical space
def UlapU(U, Uhat):
    # Laplacian U in Fourier Space
    lapUhat = -np.multiply(K2, Uhat)
    UlapU = np.multiply(ifftn(lapUhat).real, U)
    return UlapU


def mod2gradU(Uhat):    
    # Ux in Fourier Space
    Uxhat = np.multiply(KX*1j, Uhat)
    # (Ux)^2 in physical space
    Uxsquare = np.square(ifftn(Uxhat).real)

    # Uy in Fourier Space
    Uyhat = np.multiply(KY*1j, Uhat)
    # (Uy)^2 in physical space
    Uysquare = np.square(ifftn(Uyhat).real)

    # |grad U|^2 in physical space
    return (Uxsquare + Uysquare)


def F(U,Uhat):  

    if param.linear:
        return 0

    # #nonlinear terms for PDE simple case first
    if param.no_spatial:
        # Qu^2 + Au^3
        return np.multiply(U_squared(U), Q1+C1*U)

    if param.no_cubic_spatial:
        # Q1u^2 + Q2 u lapu + Q3 |grad u|^2  -u^3
        return np.multiply(U_squared(U), Q1+C1*U) + Q2*UlapU(U,Uhat) + Q3*mod2gradU(Uhat)

    if param.full_nonlinear:
        return (np.multiply(U_squared(U), Q1+C1*U) + np.multiply(Q2 + C2*U, UlapU(U,Uhat))  # Full Model PDE
                + np.multiply(Q3 + C3*U, mod2gradU(Uhat)))


def Fhat(U,Uhat):        
    return fftn(F(U,Uhat))


def RMS(U):
    return np.sqrt(1/(Nx*Ny)*np.sum(U_squared(U)))

def Fourier_Filtering(Uhat):
    for i in range(Nx):
        for j in range(Ny):
            if K[i][j] >param.k_filtering_radius:                                                     # removing all wavenumbers with k> upper bound 
                Uhat[i][j] = 0
    return Uhat

def GrowthRootsAnnulus(mu,nu):                                                                  # splitting wavenumbers into regions based off zeros of linear growth rate relation:
                                                                                                # used for visual aid when analysing solutions
    sigma_8 = sigma_0/(q**4) - mu*(q**2-3)/((q**2-1)**3) + nu*(1-3*q**2)/(q**4*(q**2-1)**3)     

    sigma_6 = (2*mu-2*sigma_8*(2*q**4+2*q**2-1)-2*sigma_0)/(3*q**2-1)           

    sigma_4 = -(3/2*sigma_6*(q**2+1)+2*sigma_8*(q**4+q**2+1))

    sigma_2 = -2*sigma_4 -3*sigma_6 - 4*sigma_8
    
    growthannulus = np.roots([sigma_8,0,sigma_6,0,sigma_4,0,sigma_2,0,sigma_0+2*max([abs(mu),abs(nu)])])
    
    growthroots = []
    for sigma in growthannulus:
        if sigma>0:
            growthroots.append(sigma)
    growthroots.sort()
    return growthroots
