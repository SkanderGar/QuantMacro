import numpy as np
import os
os.chdir('C:/Users/DELL/Desktop/Quant_macro/Pset4/hand')
import matplotlib.pyplot as plt
import Rep_agent_labor2 as ral
#The basis functions are in the class as self.func
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
hmax = 1
hmin = 0
gridk = np.linspace(kmin, kmax, n)

cheby = ral.rep_ag(para['theta'], para['beta'], para['delta'], para['kappa'], para['nu'], kmin, kmax, hmin, hmax)
New_opt, Theta = cheby.problem()
cheby.Val_pol_fun()
Vg = cheby.V(gridk)
gc = cheby.gc(gridk)
gh = cheby.gh(gridk)

f2, (ax3, ax4, ax5) = plt.subplots(1,3)
f2.set_figheight(5)
f2.set_figwidth(10)

ax3.plot(gridk, Vg, 'b', label='Value')
ax3.legend(loc = 'upper right')
ax3.set_xlabel('k')
ax3.set_ylabel('Level')
ax3.set_title('Value Function')

ax4.plot(gridk, gc, 'b', label='Consumption')
ax4.legend(loc = 'upper right')
ax4.set_xlabel('k')
ax4.set_ylabel('Level')
ax4.set_title('Policy Labor Consumption')

ax5.plot(gridk, gh, 'b', label='Labor Supply')
ax5.legend(loc = 'upper right')
ax5.set_xlabel('k')
ax5.set_ylabel('Level')
ax5.set_title('Policy Labor Supply')









