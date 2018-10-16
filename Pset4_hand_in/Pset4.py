import numpy as np
import os
os.chdir('C:/Users/DELL/Desktop/Quant_macro/Pset4/hand')
import Rep_agent as ra
import matplotlib.pyplot as plt
###parameters
para = {}
para['theta'] = 0.679
para['beta'] = 0.988
para['delta'] = 0.013
para['kappa'] = 5.24
para['nu'] = 2
para['h'] = 1

kss = (((1-para['theta'])*para['beta'])/(1-para['beta']*(1-para['delta'])))**(1/para['theta'])
n = 200
kmax = kss
kmin = 0.01*kss
gridk = np.linspace(kmin, kmax, n)
K = np.tile(gridk, (n,1)).T
Kp = K.T
Vstart = np.zeros(n)
###steady state

rep_age = ra.rep_agent(para['theta'], para['beta'], para['delta'], para['kappa'], para['nu'])
V, gk, gc = rep_age.problem(Vstart, K, Kp, type_val = 'simple')

f2, (ax3, ax4, ax5) = plt.subplots(1,3)
f2.set_figheight(5)
f2.set_figwidth(10)

ax3.plot(gridk, V, 'b', label='Value')
ax3.legend(loc = 'upper right')
ax3.set_xlabel('k')
ax3.set_ylabel('Level')
ax3.set_title('Value Function')

ax4.plot(gridk, gc, 'b', label='Consumption')
ax4.legend(loc = 'upper right')
ax4.set_xlabel('k')
ax4.set_ylabel('Level')
ax4.set_title('Policy Consumption')

ax5.plot(gridk, gk, 'b', label='Capital')
ax5.legend(loc = 'upper right')
ax5.set_xlabel('k')
ax5.set_ylabel('Level')
ax5.set_title('Policy Capital')
