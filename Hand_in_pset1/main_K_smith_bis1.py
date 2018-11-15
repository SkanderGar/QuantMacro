import numpy as np
import os
os.chdir('C:/Users/DELL/Desktop/Luis/Pset1')
import matplotlib.pyplot as plt
import K_smith as ks
#B = ks1.param_update(Guess = Beta)
Beta = np.loadtxt('parameter.txt')   
ks1 = ks.K_S(Guess = Beta, N_k = 100, N_K=10)
V_int, gk_int = ks1.Start_VFI()
Z, E = ks1.Simulation_U_Z()
dist = ks1.simulation_Ag_K(gk_int, Z, E, ret=1)
distg = ks1.simu_dist(gk_int, dist[-1,:])
distb = ks1.simu_dist(gk_int, dist[-1,:], state = 'bad')
f2, (g_ax1, g_ax2) = plt.subplots(1,2)
f2.set_figheight(5)
f2.set_figwidth(10)
g_ax1.hist(distb,bins =40, label = '7 period bad')
g_ax1.legend(loc = 'upper right')
g_ax2.hist(distg,bins =40, label = '7 period good')
g_ax2.legend(loc = 'upper right')