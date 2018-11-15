import numpy as np
import os
os.chdir('C:/Users/DELL/Desktop/Luis/Pset1')
import matplotlib.pyplot as plt
import K_smith as ks

    
ks1 = ks.K_S(N_k = 100, N_K=10)

########################
V_int, gk_int = ks1.Start_VFI()
K = ks1.grid_K[0]
K2 = ks1.grid_K[-1]
grid_k = ks1.grid_k

g1, g2, g3, g4 = gk_int[0](grid_k,K), gk_int[1](grid_k,K), gk_int[2](grid_k,K), gk_int[3](grid_k,K)

g1_2, g2_2, g3_2, g4_2 = gk_int[0](grid_k,K2), gk_int[1](grid_k,K2), gk_int[2](grid_k,K2), gk_int[3](grid_k,K2)

f, (ax1, ax2) = plt.subplots(1,2)

f.set_figheight(5)
f.set_figwidth(10)
ax1.plot(grid_k[:70], g1[:70],'b', label = 'Unemployed Good K=10')
ax1.plot(grid_k[:70], g2[:70],'r', label = 'Employed Good K=10')
ax1.plot(grid_k[:70], g1_2[:70],'g', label = 'Unemployed Good K=18')
ax1.plot(grid_k[:70], g2_2[:70],'k', label = 'Employed Good K=18')
ax1.legend(loc = 'upper right')
ax1.set_xlabel('k_t')
ax1.set_ylabel('k_t+1')
ax1.set_title('Policy Functions')
ax2.plot(grid_k[:70], g3[:70],'b', label = 'Unemployed Bad K=10')
ax2.plot(grid_k[:70], g4[:70],'r', label = 'Employed Bad K=10')
ax2.plot(grid_k[:70], g3_2[:70],'g', label = 'Unemployed Bad K=18')
ax2.plot(grid_k[:70], g4_2[:70],'k', label = 'Employed Bad K=18')
ax2.legend(loc = 'upper right')
ax2.set_xlabel('k_t')
ax2.set_ylabel('k_t+1')
ax2.set_title('Policy Functions')

#f2, (ax1_2, ax2_2) = plt.subplots(1,2)

#f2.set_figheight(5)
#f2.set_figwidth(10)
#ax1_2.plot(grid_k[:70], g1_2[:70],'b', label = 'Unemployed Good')
#ax1_2.plot(grid_k[:70], g2_2[:70],'r', label = 'Employed Good')
#ax1_2.legend(loc = 'upper right')
#ax1_2.set_xlabel('k_t')
#ax1_2.set_ylabel('k_t+1')
#ax1_2.set_title('Policy Functions')
#ax2_2.plot(grid_k[:70], g3_2[:70],'g', label = 'Unemployed Bad')
#ax2_2.plot(grid_k[:70], g4_2[:70],'k', label = 'Employed Bad')
#ax2_2.legend(loc = 'upper right')
#ax2_2.set_xlabel('k_t')
#ax2_2.set_ylabel('k_t+1')
#ax2_2.set_title('Policy Functions')
#################################
#Z, E = ks1.Simulation_U_Z()
#dist_t = ks1.simulation_Ag_K(gk_int, Z, E,ret=1)
#burn
#Z = Z[200:]
#E = E[200:]
#dist_t = dist_t[200:]
#f2, (g_ax1, g_ax2) = plt.subplots(1,2)
#f2.set_figheight(5)
#f2.set_figwidth(10)
#g_ax1.hist(dist_t[8,:],bins =30, label = '7 period bad')
#g_ax1.legend(loc = 'upper right')
#g_ax2.hist(dist_t[113,:],bins =30, label = '7 period good')
#g_ax2.legend(loc = 'upper right')



#B = ks1.param_update()