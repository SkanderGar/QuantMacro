import numpy as np
import os
os.chdir('C:/Users/DELL/Desktop/Luis/Pset2')
import matplotlib.pyplot as plt
import K_S_Labor_Q2 as ks
######initialisation######

G_max, G_min, g_max, g_min = 1.15,0.5,2.5,1.5
ub_G = G_max
lb_G = G_min
ub_g = g_max
lb_g = g_min
etaG = 0.5
count = 0
gam = (ub_g+lb_g)/2
Gam = (ub_G+lb_G)/2
target = 0.93
####################
####calibration####
###################


j=0
while True:
    print('STARTING GOOD:',j)
    ks_g = ks.K_S(N_k=50, Gam = Gam,gam=gam)
    Structure_g, L_g = ks_g.r_w_update()
    print('STARTING BAD:',j)
    ks_b = ks.K_S(N_k=50,state='bad', Gam=Gam,gam=gam)
    Structure_b, L_b = ks_b.r_w_update()
    
    L_bar = (L_g+L_b)/2
    Excess_L_bar = L_bar - target

    old_lb_G = lb_G
    old_ub_G = ub_G
    if Excess_L_bar>=0:#to much labor we want to increase desutility
        lb_G = Gam
        Gam = (1-etaG)*Gam + etaG*ub_G
    elif  Excess_L_bar<0:#to little labor deacrease desutility
        ub_G = Gam
        Gam = (1-etaG)*Gam + etaG*lb_G
    if old_lb_G==lb_G and old_ub_G == ub_G:##step conv
        etaG = etaG/2
        count = count+1

    print('RESULT CALIBRATION:',j)
    print('Error:', Excess_L_bar)
    print('Gamma:', Gam)
    print('gamma:', gam)        
        
    if count >= 4:#if step conv happens too often if will not converge
        print('not calibrated, the error is:', Excess_L_bar)
        break        
    if np.abs(Excess_L_bar)<0.01:
        break
    j=j+1       
        
        
        
        
        
        
                
                 # I put it here for clarity otherwise it should be better outside the loop
##etaG is pretty high because here it is a different kind of convergence the upper and lower bound are converging making sur that we don't miss anything
                
                    
                    
                    
                    
                    
                    
                ### this is here just to make sure that my upper and lower bound converges
                
                
                
                
                