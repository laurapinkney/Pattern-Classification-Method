# -*- coding: utf-8 -*-
"""
Created on Mon May 13 16:30:58 2024

@author: mm18lp
"""
# Version 2.0: with stars 


"""

1,2 : z stripes

3,4 : w stripes

5,6 : z hex

7,8 : w hex

9,10 : rhombs

11 : superhex

12,13 : stars

14 : superlattice

15 : fast chaos

16 : slow chaos

17 : periodic  (determined by eye)


"""

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys

sys.path.append('../')
  

###############################################################################

colourbar = False       # show colourbar True/False

separate_colourbar = True       # plot colourbar in figure or separately

zoom = False            # zoomed in for small (\mu,\nu)

wedge = False           # small wedge of (\mu,\nu) parameter space


###############################################################################

filename = '../output_files/Q1_-1.01'

patterncode = np.loadtxt(f'{filename}_patterncodenew.txt')  # load in classification data

patterncode_patch = np.loadtxt(f'{filename}_patterncodepatch.txt')

###############################################################################
    
# Lists for Pattern Classification: split into each pattern and time dependence

mu_zero, nu_zero = [],[]


mu_zstripe, nu_zstripe = [], []

mu_zstripe_defect, nu_zstripe_defect = [],[]

mu_zstripe_defect_slow, nu_zstripe_defect_slow = [],[]  # not found any examples of z_stripe fast


mu_wstripe, nu_wstripe = [], []

mu_wstripe_defect, nu_wstripe_defect = [],[]

mu_wstripe_defect_slow, nu_wstripe_defect_slow = [],[]

mu_wstripe_defect_fast, nu_wstripe_defect_fast = [],[]


mu_zhex, nu_zhex = [], []

mu_zhex_defect, nu_zhex_defect =[],[]

mu_zhex_defect_slow, nu_zhex_defect_slow =[],[]

mu_zhex_defect_fast, nu_zhex_defect_fast = [],[]


mu_whex, nu_whex = [], []

mu_whex_defect, nu_whex_defect =[],[]

mu_whex_defect_slow, nu_whex_defect_slow =[],[]

mu_whex_defect_fast, nu_whex_defect_fast = [],[]


mu_rhombs, nu_rhombs = [], []

mu_rhombs_defect, nu_rhombs_defect = [], []

mu_rhombs_defect_slow, nu_rhombs_defect_slow = [], []

mu_rhombs_defect_fast, nu_rhombs_defect_fast =[],[]


mu_superhex, nu_superhex = [], []

mu_superhex_defect, nu_superhex_defect = [],[]

mu_superhex_defect_slow, nu_superhex_defect_slow = [],[]

mu_superhex_defect_fast, nu_superhex_defect_fast = [],[]


mu_stars, nu_stars = [],[]

mu_stars_defect, nu_stars_defect = [],[]

mu_stars_defect_slow, nu_stars_defect_slow = [],[]

mu_stars_defect_fast, nu_stars_defect_fast = [],[]


mu_superlat, nu_superlat = [],[]

mu_superlat_defect, nu_superlat_defect = [],[]

mu_superlat_defect_slow, nu_superlat_defect_slow =[],[]

mu_superlat_defect_fast, nu_superlat_defect_fast =[],[]


mu_TC, nu_TC = [],[]

mu_STC, nu_STC = [],[]

mu_periodic,nu_periodic = [],[]

mu_undetermined, nu_undetermined = [],[]



r_list =[]
chi_list = []
for i in range(len(patterncode)):
    r, chi = patterncode[i][0],patterncode[i][1]
    if r.round(3) not in r_list:
        r_list.append(r.round(3))
    if chi not in chi_list:
        chi_list.append(chi)
r_list.sort(reverse=True)       # sorting from largest to smallest (order the simulations were performed in)
chi_list.sort()

paramlist = []

pattern_number = np.zeros((len(r_list),len(chi_list)))  # array for pattern classification: assigning different number to each pattern


for r in r_list:
    for chi in chi_list:
        paramlist.append([r,chi])

                       
sorted_patch = [ [] for i in range(len(patterncode))]
for i in range(len(patterncode_patch)):
    r,chi = patterncode_patch[i][0],patterncode_patch[i][1]
    r=r.round(3)
    index_number = paramlist.index([r,chi]) 
    sorted_patch[index_number].append(patterncode_patch[i])

patterncodereverse = patterncode[::-1] # reversing array to use classification from latest simulation (useful if a simulation was rerun for longer if not converged)

no_repeats = []

for i in range(len(patterncode)):
    r, chi, k1_peaks, kq_peaks, k1_fuzzy, kq_fuzzy, Equilibrium, Slow, Fast, Small_Vars, Small_Spatial_Change, Large_Spatial_Change = patterncodereverse[i]
    r = r.round(3)
    if [r,chi] not in no_repeats:                   # ignores multiple values of r,chi from non converged runs
        no_repeats.append([r,chi])
        
        # insert any manually set pattern classifcations here
        

        if Equilibrium:                                                        # Equilibria
    
            if k1_peaks ==0 and kq_peaks ==0:                                  # no pattern
                mu_zero.append(r*np.cos(chi*math.pi/180))
                nu_zero.append(r*np.sin(chi*math.pi/180))
                
                pattern_number[r_list.index(r)][chi_list.index(chi)] = 0
            
            elif k1_fuzzy == False and kq_fuzzy == False:                        # perfect pattern: no defects
            
                if k1_peaks ==2 and kq_peaks ==0:                              # z stripes
                    mu_zstripe.append(r*np.cos(chi*math.pi/180))
                    nu_zstripe.append(r*np.sin(chi*math.pi/180))
                    
                    pattern_number[r_list.index(r)][chi_list.index(chi)] = 1
                    
                elif k1_peaks ==0 and kq_peaks ==2:                            # w stripes
                    mu_wstripe.append(r*np.cos(chi*math.pi/180))
                    nu_wstripe.append(r*np.sin(chi*math.pi/180))
                    
                    pattern_number[r_list.index(r)][chi_list.index(chi)] = 3
                        
                elif k1_peaks ==6 and kq_peaks ==0:                            # z-hex
                    mu_zhex.append(r*np.cos(chi*math.pi/180))
                    nu_zhex.append(r*np.sin(chi*math.pi/180))
                    
                    pattern_number[r_list.index(r)][chi_list.index(chi)] = 5
                    
                elif k1_peaks ==0 and kq_peaks ==6:                            # w-hex
                    mu_whex.append(r*np.cos(chi*math.pi/180))
                    nu_whex.append(r*np.sin(chi*math.pi/180))
                    
                    pattern_number[r_list.index(r)][chi_list.index(chi)] = 7
                    
                elif k1_peaks ==4 and kq_peaks ==2:                            # rhombs
                    mu_rhombs.append(r*np.cos(chi*math.pi/180))
                    nu_rhombs.append(r*np.sin(chi*math.pi/180))
                 
                    pattern_number[r_list.index(r)][chi_list.index(chi)] = 9
        
                elif (k1_peaks ==12 or k1_peaks==6) and kq_peaks ==6:          # stars or superhex: from experience stars more likely so setting as stars. 
                    mu_stars.append(r*np.cos(chi*math.pi/180))                 # manually override if needed  
                    nu_stars.append(r*np.sin(chi*math.pi/180))
                    
                    pattern_number[r_list.index(r)][chi_list.index(chi)] = 12

                
                elif k1_peaks >0 and kq_peaks ==6:                             # superlattice: other undetermined superlattice pattern
                    mu_superlat.append(r*np.cos(chi*math.pi/180))
                    nu_superlat.append(r*np.sin(chi*math.pi/180))
                    
                    pattern_number[r_list.index(r)][chi_list.index(chi)] = 14
 
                else:
                    print(str('Missing Values'),patterncodereverse[i],paramlist.index([r,chi])+1) # print values for any simulations not satisfying these conditions
            
            
            elif kq_fuzzy == False:                                            # fuzziness on k=1 circle only

                if k1_peaks ==2 and kq_peaks ==0 and k1_fuzzy:                 # z stripe defect
                    mu_zstripe_defect.append(r*np.cos(chi*math.pi/180)) 
                    nu_zstripe_defect.append(r*np.sin(chi*math.pi/180))
                    
                    pattern_number[r_list.index(r)][chi_list.index(chi)] = 2  

                elif k1_peaks ==6 and kq_peaks ==0:                            # zhex with defect
                        mu_zhex_defect.append(r*np.cos(chi*math.pi/180))
                        nu_zhex_defect.append(r*np.sin(chi*math.pi/180))
                        
                        pattern_number[r_list.index(r)][chi_list.index(chi)] = 6                                         
            
                elif k1_peaks ==0 and kq_peaks ==6:                            # w-hex: for q=1/\sqrt{7} can have fuzziness but no peaks
                    mu_whex.append(r*np.cos(chi*math.pi/180))
                    nu_whex.append(r*np.sin(chi*math.pi/180))
                    
                    pattern_number[r_list.index(r)][chi_list.index(chi)] = 7
                    
                elif k1_peaks ==0 and kq_peaks ==2:                            # rhombs: often k1 contributions are too small to be peaks but looks like rhombs not stripes
                    mu_rhombs.append(r*np.cos(chi*math.pi/180))
                    nu_rhombs.append(r*np.sin(chi*math.pi/180))
                 
                    pattern_number[r_list.index(r)][chi_list.index(chi)] = 9
                    

                elif (k1_peaks==12 or k1_peaks==6) and kq_peaks ==6:           # stars: using P1=6 (peaks on k=1) as well as P1=12 as often only one set of amplitudes
                    mu_stars.append(r*np.cos(chi*math.pi/180))                 #        are large enough to be peaks, the other gets classed as 'fuzziness'
                    nu_stars.append(r*np.sin(chi*math.pi/180))
                    
                    pattern_number[r_list.index(r)][chi_list.index(chi)] = 12
             
                elif k1_peaks>0 and kq_peaks ==6 and k1_fuzzy:                  # superlattice with defect
                    mu_superlat_defect.append(r*np.cos(chi*math.pi/180))
                    nu_superlat_defect.append(r*np.sin(chi*math.pi/180))
                    
                    pattern_number[r_list.index(r)][chi_list.index(chi)] = 14    
                    
                else:
                    print(str('Missing Values'),patterncodereverse[i],paramlist.index([r,chi])+1)
        
            # potential fuzziness on both circles
            
            elif k1_peaks ==6 and kq_peaks ==0:                                # zhex with defect
                    mu_zhex_defect.append(r*np.cos(chi*math.pi/180))
                    nu_zhex_defect.append(r*np.sin(chi*math.pi/180))
                    
                    pattern_number[r_list.index(r)][chi_list.index(chi)] = 6
                    
            elif k1_peaks ==0 and (kq_peaks ==2 or kq_peaks ==4) and k1_fuzzy and kq_fuzzy: # rhombs with defect, often see multiple orientations giving Pq=4
                mu_rhombs_defect.append(r*np.cos(chi*math.pi/180))
                nu_rhombs_defect.append(r*np.sin(chi*math.pi/180))
             
                pattern_number[r_list.index(r)][chi_list.index(chi)] = 10
                
                
            elif k1_peaks ==0 and (kq_peaks ==2 or kq_peaks ==4) and kq_fuzzy and k1_fuzzy==False:      # wstripe with defect (no fuzziness on k=1 circle)
                mu_wstripe_defect.append(r*np.cos(chi*math.pi/180)) 
                nu_wstripe_defect.append(r*np.sin(chi*math.pi/180))
                
                pattern_number[r_list.index(r)][chi_list.index(chi)] = 4
                
                
            elif k1_peaks ==0 and (kq_peaks ==6 or kq_peaks ==8 or kq_peaks ==10 or kq_peaks==12 or kq_peaks%2==1): # cases to use patch analysis
                index_number = paramlist.index([r,chi])
                patch_data = sorted_patch[index_number][-3] # first patch (taking from end in case of multiple entries from non-converged runs)
                k1_peaks, kq_peaks, k1_fuzzy, kq_fuzzy = patch_data[3:7]
                if kq_peaks == 2:
                    mu_wstripe_defect.append(r*np.cos(chi*math.pi/180))   # w stripe defect
                    nu_wstripe_defect.append(r*np.sin(chi*math.pi/180))
                    
                    pattern_number[r_list.index(r)][chi_list.index(chi)] = 4
                    
                elif kq_peaks == 6:                                            # whex defect
                    mu_whex_defect.append(r*np.cos(chi*math.pi/180)) 
                    nu_whex_defect.append(r*np.sin(chi*math.pi/180))
                    
                    pattern_number[r_list.index(r)][chi_list.index(chi)] = 8
                    
                else:                                                          # try second patch when first fails
                    patch_data = sorted_patch[index_number][-2] # second patch only
                    k1_peaks, kq_peaks, k1_fuzzy, kq_fuzzy = patch_data[3:7]
                    
                    if kq_peaks == 2:
                        mu_wstripe_defect.append(r*np.cos(chi*math.pi/180))# w stripe defect
                        nu_wstripe_defect.append(r*np.sin(chi*math.pi/180))
                        
                        pattern_number[r_list.index(r)][chi_list.index(chi)] = 4
                        
                    elif kq_peaks == 6:                                        # whex defect
                        mu_whex_defect.append(r*np.cos(chi*math.pi/180)) 
                        nu_whex_defect.append(r*np.sin(chi*math.pi/180))
                        
                        pattern_number[r_list.index(r)][chi_list.index(chi)] = 8
                    
                        
                    else:                                                       # try third patch if first two fail
                        patch_data = sorted_patch[index_number][-1] # third patch only
                        k1_peaks, kq_peaks, k1_fuzzy, kq_fuzzy = patch_data[3:7]
                        
                        if kq_peaks == 2:
                            mu_wstripe_defect.append(r*np.cos(chi*math.pi/180))# w stripe defect
                            nu_wstripe_defect.append(r*np.sin(chi*math.pi/180))
                            
                            pattern_number[r_list.index(r)][chi_list.index(chi)] = 4
                            
                        elif kq_peaks == 6:                                        # whex defect
                            mu_whex_defect.append(r*np.cos(chi*math.pi/180)) 
                            nu_whex_defect.append(r*np.sin(chi*math.pi/180))
                            
                            pattern_number[r_list.index(r)][chi_list.index(chi)] = 8
                            
                        else:
                            print(str('Missing Values: Patch'),patterncodereverse[i],paramlist.index([r,chi])+1) # print values for any simulations not satisfying these conditions
    
                    
                    
            
            elif (k1_peaks==12 or k1_peaks ==6) and kq_peaks ==6:              # stars or superhex with defect
                mu_stars_defect.append(r*np.cos(chi*math.pi/180))
                nu_stars_defect.append(r*np.sin(chi*math.pi/180))
                
                pattern_number[r_list.index(r)][chi_list.index(chi)] = 13 
                
            elif k1_peaks>0 and kq_peaks ==6 and (k1_fuzzy or kq_fuzzy):       # superlattice with defect
                mu_superlat_defect.append(r*np.cos(chi*math.pi/180))
                nu_superlat_defect.append(r*np.sin(chi*math.pi/180))
                
                pattern_number[r_list.index(r)][chi_list.index(chi)] = 14

                
            else:                                                              # check for missing equilibria
                print(str('Missing Values'),patterncodereverse[i],paramlist.index([r,chi])+1)
    
    
        elif Slow: # always has SV == True and SSC == LSC == False
        
            if k1_peaks ==0 and (kq_peaks ==2 or kq_peaks ==4): # w stripe cases
            
                if kq_peaks ==2 and kq_fuzzy== False and k1_fuzzy== False:      # near converged w stripe: sometimes find this
                    mu_wstripe.append(r*np.cos(chi*math.pi/180))
                    nu_wstripe.append(r*np.sin(chi*math.pi/180))
                    
                    pattern_number[r_list.index(r)][chi_list.index(chi)] = 3
                
                elif kq_fuzzy and k1_fuzzy==False:
                    mu_wstripe_defect_slow.append(r*np.cos(chi*math.pi/180))   # w stripe defect
                    nu_wstripe_defect_slow.append(r*np.sin(chi*math.pi/180))
                    
                    pattern_number[r_list.index(r)][chi_list.index(chi)] = 4 
                    
                else:
                    mu_rhombs_defect_slow.append(r*np.cos(chi*math.pi/180))    # rhombs with defects
                    nu_rhombs_defect_slow.append(r*np.sin(chi*math.pi/180))
                 
                    pattern_number[r_list.index(r)][chi_list.index(chi)] = 10
            
            elif k1_peaks ==2 and kq_peaks ==0 and k1_fuzzy:
                mu_zstripe_defect_slow.append(r*np.cos(chi*math.pi/180))       # z stripe defect
                nu_zstripe_defect_slow.append(r*np.sin(chi*math.pi/180))
                
                pattern_number[r_list.index(r)][chi_list.index(chi)] = 2 
                
            elif k1_peaks ==2 and kq_peaks ==0 and k1_fuzzy== False and kq_fuzzy==False:  # near converged z stripe: sometimes find this.
                mu_zstripe.append(r*np.cos(chi*math.pi/180))
                nu_zstripe.append(r*np.sin(chi*math.pi/180))
                
                pattern_number[r_list.index(r)][chi_list.index(chi)] = 1
                
            elif k1_peaks ==6 and kq_peaks ==0 and k1_fuzzy == False and kq_fuzzy == False: # near converfed z hex: sometimes find this
                mu_zhex.append(r*np.cos(chi*math.pi/180))
                nu_zhex.append(r*np.sin(chi*math.pi/180))
                
                pattern_number[r_list.index(r)][chi_list.index(chi)] = 5
                
            elif k1_peaks ==6 and (kq_peaks ==0):                               # zhex with defects
                mu_zhex_defect_slow.append(r*np.cos(chi*math.pi/180))
                nu_zhex_defect_slow.append(r*np.sin(chi*math.pi/180))
                
                pattern_number[r_list.index(r)][chi_list.index(chi)] = 6
    
            elif k1_peaks ==0 and (kq_peaks ==6 or kq_peaks ==8 or kq_peaks ==10 or kq_peaks==12 or kq_peaks%2==1): #odd number of peaks 
                index_number = paramlist.index([r,chi])
                patch_data = sorted_patch[index_number][-3] # first patch (taking from end in case of multiple entries from non-converged runs)
                k1_peaks, kq_peaks, k1_fuzzy, kq_fuzzy = patch_data[3:7]
                if kq_peaks == 2:
                    mu_wstripe_defect_slow.append(r*np.cos(chi*math.pi/180))   # w stripe defect
                    nu_wstripe_defect_slow.append(r*np.sin(chi*math.pi/180))
                    
                    pattern_number[r_list.index(r)][chi_list.index(chi)] = 4
                    
                elif kq_peaks == 6:                                            # whex defect
                    mu_whex_defect_slow.append(r*np.cos(chi*math.pi/180)) 
                    nu_whex_defect_slow.append(r*np.sin(chi*math.pi/180))
                    
                    pattern_number[r_list.index(r)][chi_list.index(chi)] = 8
                    
                else:                                                          # try second patch when first fails
                    patch_data = sorted_patch[index_number][-2] # second patch only
                    k1_peaks, kq_peaks, k1_fuzzy, kq_fuzzy = patch_data[3:7]
                    
                    if kq_peaks == 2:
                        mu_wstripe_defect_slow.append(r*np.cos(chi*math.pi/180))# w stripe defect
                        nu_wstripe_defect_slow.append(r*np.sin(chi*math.pi/180))
                        
                        pattern_number[r_list.index(r)][chi_list.index(chi)] = 4
                        
                    elif kq_peaks == 6:                                        # whex defect
                        mu_whex_defect_slow.append(r*np.cos(chi*math.pi/180)) 
                        nu_whex_defect_slow.append(r*np.sin(chi*math.pi/180))
                        
                        pattern_number[r_list.index(r)][chi_list.index(chi)] = 8
                    
                        
                    else:
                        patch_data = sorted_patch[index_number][-1] # third patch only
                        k1_peaks, kq_peaks, k1_fuzzy, kq_fuzzy = patch_data[3:7]
                        
                        if kq_peaks == 2:
                            mu_wstripe_defect_slow.append(r*np.cos(chi*math.pi/180))# w stripe defect
                            nu_wstripe_defect_slow.append(r*np.sin(chi*math.pi/180))
                            
                            pattern_number[r_list.index(r)][chi_list.index(chi)] = 4
                            
                        elif kq_peaks == 6:                                        # whex defect
                            mu_whex_defect_slow.append(r*np.cos(chi*math.pi/180)) 
                            nu_whex_defect_slow.append(r*np.sin(chi*math.pi/180))
                            
                            pattern_number[r_list.index(r)][chi_list.index(chi)] = 8
                            
                        else:
                            print(str('Missing Values: Patch'),patterncodereverse[i],paramlist.index([r,chi])+1) # print values for any simulations not satisfying these conditions
                    
        
            elif k1_peaks ==4 and (kq_peaks ==2 or kq_peaks ==4) and (k1_fuzzy or kq_fuzzy):
                mu_rhombs_defect_slow.append(r*np.cos(chi*math.pi/180))              # rhombs with defects
                nu_rhombs_defect_slow.append(r*np.sin(chi*math.pi/180))
             
                pattern_number[r_list.index(r)][chi_list.index(chi)] = 10
                    
            elif (k1_peaks ==12 or k1_peaks ==6) and kq_peaks ==6 and (k1_fuzzy or kq_fuzzy):
                mu_stars_defect_slow.append(r*np.cos(chi*math.pi/180))          # stars (or superhex) with defects
                nu_stars_defect_slow.append(r*np.sin(chi*math.pi/180))
                
                pattern_number[r_list.index(r)][chi_list.index(chi)] = 13 
              
                    
            elif k1_peaks>0 and kq_peaks ==6:                                   # superlattice with defects
                mu_superlat_defect_slow.append(r*np.cos(chi*math.pi/180))
                nu_superlat_defect_slow.append(r*np.sin(chi*math.pi/180))
                
                pattern_number[r_list.index(r)][chi_list.index(chi)] = 14

        
            else:
                print(str('Missing Values'),patterncodereverse[i],paramlist.index([r,chi])+1)  # check for missing values 
                                                                                       
        
        else: #Fast
            if kq_peaks>0:
                if Small_Spatial_Change and Small_Vars == False:               # Slower spatially changing chaos
                    mu_TC.append(r*np.cos(chi*math.pi/180))
                    nu_TC.append(r*np.sin(chi*math.pi/180))   
                    
                    pattern_number[r_list.index(r)][chi_list.index(chi)] = 16
                    
                elif Large_Spatial_Change and Small_Vars == False:             # Faster spatially changing chaos
                    mu_STC.append(r*np.cos(chi*math.pi/180))
                    nu_STC.append(r*np.sin(chi*math.pi/180))   
                    
                    pattern_number[r_list.index(r)][chi_list.index(chi)] = 15
                
                
                elif k1_peaks ==0 and (kq_peaks ==2 or kq_peaks ==4) and kq_fuzzy and k1_fuzzy==False:     
                
                    mu_wstripe_defect_fast.append(r*np.cos(chi*math.pi/180))    # w stripe defect
                    nu_wstripe_defect_fast.append(r*np.sin(chi*math.pi/180))
                    
                    pattern_number[r_list.index(r)][chi_list.index(chi)] = 4
                    
                elif (k1_peaks ==12 or k1_peaks ==6) and kq_peaks ==6 and (k1_fuzzy or kq_fuzzy):
                    mu_stars_defect_fast.append(r*np.cos(chi*math.pi/180))      # stars (or superhex) with defects
                    nu_stars_defect_fast.append(r*np.sin(chi*math.pi/180))
                    
                    pattern_number[r_list.index(r)][chi_list.index(chi)] = 13 
                    
                elif k1_peaks >0 and kq_peaks ==6:                             # superlat defect
                    mu_superlat_defect_fast.append(r*np.cos(chi*math.pi/180))
                    nu_superlat_defect_fast.append(r*np.sin(chi*math.pi/180))
                    
                    pattern_number[r_list.index(r)][chi_list.index(chi)] = 14
                   
                elif k1_peaks ==0 and (kq_peaks==6 or kq_peaks ==8 or kq_peaks==10 or kq_peaks ==12 or kq_peaks%2==1):
                    index_number = paramlist.index([r,chi])
                    patch_data = sorted_patch[index_number][-3] # first patch (taking from end in case of multiple entries from non-converged runs)
                    k1_peaks, kq_peaks, k1_fuzzy, kq_fuzzy = patch_data[3:7]
                    if kq_peaks == 2:
                        mu_wstripe_defect_fast.append(r*np.cos(chi*math.pi/180))   # w stripe defect
                        nu_wstripe_defect_fast.append(r*np.sin(chi*math.pi/180))
                        
                        pattern_number[r_list.index(r)][chi_list.index(chi)] = 4
                        
                    elif kq_peaks == 6:                                            # whex defect
                        mu_whex_defect_fast.append(r*np.cos(chi*math.pi/180)) 
                        nu_whex_defect_fast.append(r*np.sin(chi*math.pi/180))
                        
                        pattern_number[r_list.index(r)][chi_list.index(chi)] = 8
                        
                    else:                                                          # try second patch when first fails
                        patch_data = sorted_patch[index_number][-2] # second patch only
                        k1_peaks, kq_peaks, k1_fuzzy, kq_fuzzy = patch_data[3:7]
                        
                        if kq_peaks == 2:
                            mu_wstripe_defect_fast.append(r*np.cos(chi*math.pi/180))# w stripe defect
                            nu_wstripe_defect_fast.append(r*np.sin(chi*math.pi/180))
                            
                            pattern_number[r_list.index(r)][chi_list.index(chi)] = 4
                            
                        elif kq_peaks == 6:                                        # whex defect
                            mu_whex_defect_fast.append(r*np.cos(chi*math.pi/180)) 
                            nu_whex_defect_fast.append(r*np.sin(chi*math.pi/180))
                            
                            pattern_number[r_list.index(r)][chi_list.index(chi)] = 8
                        
                            
                        else:
                            patch_data = sorted_patch[index_number][-1] # third patch only
                            k1_peaks, kq_peaks, k1_fuzzy, kq_fuzzy = patch_data[3:7]
                            
                            if kq_peaks == 2:
                                mu_wstripe_defect_fast.append(r*np.cos(chi*math.pi/180))# w stripe defect
                                nu_wstripe_defect_fast.append(r*np.sin(chi*math.pi/180))
                                
                                pattern_number[r_list.index(r)][chi_list.index(chi)] = 4
                                
                            elif kq_peaks == 6:                                        # whex defect
                                mu_whex_defect_fast.append(r*np.cos(chi*math.pi/180)) 
                                nu_whex_defect_fast.append(r*np.sin(chi*math.pi/180))
                                
                                pattern_number[r_list.index(r)][chi_list.index(chi)] = 8
                                
                            else:
                                print(str('Missing Values: Patch'),patterncodereverse[i],paramlist.index([r,chi])+1) # print values for any simulations not satisfying these conditions
                        
                        
                elif (k1_peaks ==0 or k1_peaks ==4) and (kq_peaks ==2 or kq_peaks ==4) and k1_fuzzy and kq_fuzzy:
                    mu_rhombs_defect_fast.append(r*np.cos(chi*math.pi/180))    # rhombs with defects
                    nu_rhombs_defect_fast.append(r*np.sin(chi*math.pi/180))
                 
                    pattern_number[r_list.index(r)][chi_list.index(chi)] = 10  
         
                
                else:
                    print(str('Missing Values'),patterncodereverse[i],paramlist.index([r,chi])+1) # check for missing values
            
            elif k1_peaks ==6:                                                 # z hex with defect
                mu_zhex_defect_fast.append(r*np.cos(chi*math.pi/180)) 
                nu_zhex_defect_fast.append(r*np.sin(chi*math.pi/180))
                
                pattern_number[r_list.index(r)][chi_list.index(chi)] = 6 
            
            
            else:
                print(str('Missing Values'),patterncodereverse[i],paramlist.index([r,chi])+1)         # check for missing values
                





for i in range(len(r_list)):
    for j in range(len(chi_list)):
        if pattern_number[i][j]>10 and pattern_number[i][j]<14:
            r = r_list[i]
            chi = chi_list[j]
            print(str('Stars or Superhex'),r,chi,paramlist.index([r,chi])+1)   # manually check if stars or superhex 
        elif pattern_number[i][j]==14:
            r = r_list[i]
            chi = chi_list[j]
            print(str('Other Superlattice'),r,chi,paramlist.index([r,chi])+1)  # manually check superlattice

                    
print('###################Complete###########################')      


R, CHI = np.meshgrid(np.array(r_list),np.array(chi_list),indexing='ij')

mu_all = np.multiply(R,np.cos(CHI*math.pi/180))
nu_all = np.multiply(R,np.sin(CHI*math.pi/180))
   

# colourmap for each equilibrium pattern
cmaps =  mpl.colors.ListedColormap(['gold','khaki','limegreen','lightgreen','deepskyblue','lightskyblue','red','tomato','mediumaquamarine','aquamarine','mediumblue','0.6','silver','0.9','darkorchid','orchid','lightpink'])
  
###############################################################################

# Plotting edge for diagram
 
mu_edge = np.copy(r_list)
nu_edge = np.copy(mu_edge)

patterncode_mu = np.zeros(np.shape(mu_edge))
patterncode_nu = np.zeros(np.shape(nu_edge))  
    
for i in range(len(patterncode_mu)):
    r = abs(mu_edge[i])
    patterncode_mu[i] = pattern_number[r_list.index(r),-1]
    patterncode_nu[i] = pattern_number[r_list.index(r),0]
  
mu_edge_2D = np.array([mu_edge,mu_edge])
nu_edge_2D = np.array([nu_edge,nu_edge])    

mu_range_2 = [r*np.cos(-85/180*math.pi) for r in r_list]
mu_range_1 = [ r*np.cos(-88/180*math.pi) for r in r_list]

nu_range_1 = [r*np.sin(175/180*math.pi) for r in r_list]
nu_range_2 = [ r*np.sin(178.5/180*math.pi) for r in r_list]
  
###############################################################################

# point for equilibria
# ^ for travelling solutions
# x for chaos
    
plt.figure('Bifurcation Contour with Point',figsize=(7,6))
plt.xlabel('$\mu$')
plt.ylabel('$\\nu$')
axes = plt.gca()
axes.xaxis.label.set_size(20)
axes.yaxis.label.set_size(20)
if not wedge:
    if zoom:
        axes.set_xlim([-0.045,0.045])
        axes.set_ylim([-0.028,0.06])
        axes.set(xticks=[-0.04,-0.02,0,0.02,0.04],yticks=[-0.02,0,0.02,0.04,0.06])
    else:
        axes.set_xlim([-0.53,0.53])
        axes.set_ylim([-0.53,0.53])
        axes.set(xticks=[-0.5,0,0.5],yticks=[-0.5,0,0.5])
axes.set_aspect('equal','box')
axes.tick_params(axis='both',labelsize=18)
plt.plot(mu_zstripe,nu_zstripe,'k.',markersize=3)
plt.plot(mu_zstripe_defect,nu_zstripe_defect,'k.',markersize=3)
plt.plot(mu_zstripe_defect_slow,nu_zstripe_defect_slow,'k^',markersize=3)
plt.plot(mu_wstripe,nu_wstripe,'k.',markersize=3)
plt.plot(mu_rhombs,nu_rhombs,'k.',markersize=3)
plt.plot(mu_rhombs_defect,nu_rhombs_defect,'k.',markersize=3)
plt.plot(mu_wstripe_defect,nu_wstripe_defect,'k.',markersize=3)
plt.plot(mu_rhombs_defect_slow,nu_rhombs_defect_slow,'k^',markersize=3)
plt.plot(mu_wstripe_defect_slow,nu_wstripe_defect_slow,'k^',markersize=3)
plt.plot(mu_whex_defect,nu_whex_defect,'k.',markersize=3)
plt.plot(mu_whex_defect_slow,nu_whex_defect_slow,'k^',markersize=3)
plt.plot(mu_whex_defect_fast,nu_whex_defect_fast,'k',marker='+',linestyle='None',markersize=3)
plt.plot(mu_zhex_defect_fast,nu_zhex_defect_fast,'k',marker='+',linestyle='None',markersize=3)
plt.plot(mu_rhombs_defect_fast,nu_rhombs_defect_fast,'k',marker='+',linestyle='None',markersize=3)
plt.plot(mu_wstripe_defect_fast,nu_wstripe_defect_fast,'k',marker='+',linestyle='None',markersize=3)
plt.plot(mu_superhex_defect_fast,nu_superhex_defect_fast,'k',marker='+',linestyle='None',markersize=3)
plt.plot(mu_superlat_defect_fast,nu_superlat_defect_fast,'k',marker='+',linestyle='None',markersize=3)
plt.plot(mu_zhex_defect,nu_zhex_defect,'k.',markersize=3)
plt.plot(mu_zhex_defect_slow,nu_zhex_defect_slow,'k^',markersize=3)
plt.plot(mu_zhex,nu_zhex,'k.',markersize=3)
plt.plot(mu_whex,nu_whex,color='k',linestyle='None',marker='.',markersize=3)
plt.plot(mu_STC,nu_STC,'kx',markersize=3)
plt.plot(mu_TC,nu_TC,'k',marker = 'x',linestyle='None',markersize=3)
plt.plot(mu_superhex,nu_superhex,'k',marker = '.',linestyle='None',markersize=3)
plt.plot(mu_superhex_defect,nu_superhex_defect,'k',marker = '.',linestyle='None',markersize=3)
plt.plot(mu_superhex_defect_slow,nu_superhex_defect_slow,'k',marker = '^',linestyle='None',markersize=3)
plt.plot(mu_superlat_defect,nu_superlat_defect,'k',marker = '.',linestyle='None',markersize=3)
plt.plot(mu_superlat_defect_slow,nu_superlat_defect_slow,'k',marker = '^',linestyle='None',markersize=3)
plt.plot(mu_superlat,nu_superlat,'k',marker = '.',linestyle='None',markersize=3)
plt.plot(mu_periodic,nu_periodic,'k',marker='x',linestyle='None',markersize=3)
plt.plot(mu_zero,nu_zero,'k',marker = '.',linestyle='None',markersize=3)
plt.plot(mu_stars,nu_stars,'k',marker = '.',linestyle='None',markersize=3)
plt.plot(mu_stars_defect,nu_stars_defect,'k',marker = '.',linestyle='None',markersize=3)
plt.plot(mu_stars_defect_slow,nu_stars_defect_slow,'k',marker = '^',linestyle='None',markersize=3)
plt.plot(mu_stars_defect_fast,nu_stars_defect_fast,'k',marker='+',linestyle='None',markersize=3)
plt.pcolormesh(np.array(mu_all),np.array(nu_all),np.array(pattern_number),vmin=0.5,vmax=17.5,cmap =cmaps)


if not wedge: # adding edges if full diagram is plotted
    circle1 = plt.Circle((0, 0), 0.525, edgecolor='k',fill=None,linewidth=1.2)
    axes.add_patch(circle1)
    plt.plot([-0.525,0],[0,0],'k',linewidth=1.2)
    plt.plot([0,0],[-0.525,0],'k',linewidth=1.2)
    plt.text(-0.42,-0.23,'Trivial Solution \n Stable',fontsize=15,ma='center',wrap=True)
    plt.pcolormesh(-mu_edge_2D,np.array([nu_range_1,nu_range_2]),np.array([patterncode_mu,patterncode_mu]),vmin=0.5,vmax=17.5,cmap =cmaps,shading = 'auto')
    plt.pcolormesh(np.array([mu_range_1,mu_range_2]),-nu_edge_2D,np.array([patterncode_nu,patterncode_nu]),vmin=0.5,vmax=17.5,cmap =cmaps,shading = 'auto')



if colourbar: # if adding colourbar to plot
    if separate_colourbar:          # if plotting a separate colourbar
        fig,ax = plt.subplots(figsize=(18,1.2))
        cax = ax.imshow(np.random.random((11,1))*18,vmin=0.5,vmax=17.5,cmap=cmaps)
        plt.gca().set_visible(False)
        cbar = fig.colorbar(cax,orientation='horizontal',shrink=0.8, aspect=50)
        
        vlines = [0.5,2.5,4.5,6.5,8.5,10.5,11.5,13.5,14.5,15.5,16.5,17.5]
        for v in vlines:
            cbar.ax.plot([v,v],[-1.5,1.5],color='black',linewidth=2,clip_on=False)
        
        # Tick positions and labels
        ticks = [1.5, 3.5, 5.5, 7.5, 9.5, 11, 12.5, 14, 15, 16, 17]
        labels = ['$z$-stripe', '$w$-stripe', '$z$-hex', '$w$-hex', 'rhombs', 'superhex',
                  'stars', 'super- \n lattice', 'fast \n chaos', 'slow \n chaos', '(near) \n periodic']
        
        # Hide default ticks
        cbar.ax.set_xticks([])
        cbar.ax.set_xticklabels([])
        
        # Alternate placement: top and bottom
        for i, (tick, label) in enumerate(zip(ticks, labels)):
            if i % 2 == 0:
                cbar.ax.text(tick, 1.5, label, ha='center', va='bottom', fontsize=18)  # Top
            else:
                cbar.ax.text(tick, -1.5, label, ha='center', va='top', fontsize=18)    # Bottom
        
                plt.tight_layout()
    else:
        plt.figure('Bifurcation Contour with Point')
        cbar = plt.colorbar(ticks=[1.5,3.5,5.5,7.5,9.5,11,12.5,14,15,16,17])
        cbar.ax.set_yticklabels(['$z$-stripe', '$w$-stripe','$z$-hex',  '$w$-hex', 'rhombs', 'superhex','stars',  'super- \n lattice', 'fast \n chaos', 'slow \n chaos', '(near) \n periodic'])
        


plt.figure('Bifurcation Contour with Point') 
plt.tight_layout()
plt.show()           
            
