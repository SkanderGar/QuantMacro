import numpy as np
import os
os.chdir('C:/Users/DELL/Desktop/Luis/Pset1')
import matplotlib.pyplot as plt
import K_smith as ks

#####best fit
Beta = np.loadtxt('parameter.txt')   
ks1 = ks.K_S(Guess = Beta, N_k = 100, N_K=40)
V_int, gk_int = ks1.Start_VFI()

ks2 = ks.K_S(N_k = 100, N_K=40)
V_int2, gk_int2 = ks2.Start_VFI()
K = 17
grid_k = ks1.grid_k
f2, g_ax1 = plt.subplots(1,1)
f2.set_figheight(5)
f2.set_figwidth(10)

g_ax1.plot(grid_k, V_int[0](grid_k,K), 'b' ,label = 'Not Updating')
#g_ax1.legend(loc = 'upper right')
g_ax1.plot(grid_k, V_int2[0](grid_k,K), 'r', label = 'Updating')
g_ax1.legend(loc = 'upper right')

#Z, E = ks1.Simulation_U_Z()
#k_dist, K = ks1.Simu_2ag(gk_int, gk_int2, Z, E, ret=1)
#k_m = np.mean(k_dist, axis = 0)
#K_m = np.mean(K)
#f2, (g_ax1, g_ax2) = plt.subplots(1,2)
#f2.set_figheight(5)
#f2.set_figwidth(10)

#g_ax1.hist(k_m[:1000], bins =40, label = 'Not Updating')
#g_ax1.legend(loc = 'upper right')
#g_ax2.hist(k_m[1000:], bins =40, label = 'Updating')
#g_ax2.legend(loc = 'upper right')


