import numpy as np
import os
os.chdir('C:/Users/DELL/Desktop/Luis/Pset2')
import matplotlib.pyplot as plt
import K_S_Labor_Q3 as ks

ks1 = ks.K_S(N_k = 50)
Z, E = ks1.Simulation_U_Z()
ki, ni, r_v, w_v = ks1.simulation_Ag_K(gk_Mat, gn_Mat, Z, E, ret = 1)
K = np.mean(ki, axis=1)
N = np.mean(ni, axis=1)
alpha = ks1.alpha
z = ks1.z
r = np.vectorize(lambda K,N,g: alpha*z[g]*(K/N)**(alpha-1))
w = np.vectorize(lambda K,N,g: (1-alpha)*z[g]*(K/N)**alpha)
r_v_true = r(K,N,Z)
w_v_true = w(K,N,Z)

f, (ax1, ax2) = plt.subplots(1,2)
f.set_figheight(5)
f.set_figwidth(10)
ax1.plot(r_v, 'b', label = 'Forecast')
ax1.plot(r_v_true[1:],'r', label = 'True')
ax1.legend(loc = 'upper right')
ax1.set_xlabel('Time')
ax1.set_ylabel('Level')
ax1.set_title('Interest rate')
ax2.plot(w_v,'b', label = 'Forecast')
ax2.plot(w_v_true,'r', label = 'True')
ax2.legend(loc = 'upper right')
ax2.set_xlabel('Time')
ax2.set_ylabel('Level')
ax2.set_title('Wages')