
import numpy as np
import os
import matplotlib.pyplot as plt
os.chdir('C:/Users/DELL/Desktop/Quant_macro/Pset4')
import Rep_again as ral
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
Z = np.array([0.99, 1.01])
gridk = np.linspace(kmin, kmax, n)
zm = Z[0]*np.ones(len(gridk))
zp = Z[1]*np.ones(len(gridk))
gridk = np.hstack((gridk,gridk))
Zgrid = np.hstack((zm,zp))

cheby = ral.Rep_age_shock(para['theta'], para['beta'], para['delta'], para['kappa'], para['nu'], kmin, kmax, hmin, hmax)
Zgrid_cheb = 2*(Zgrid-cheby.Z[0])/(cheby.Z[1]-cheby.Z[0]) -1
gridk_cheb = 2*(gridk-cheby.kmin)/(cheby.kmax-cheby.kmin) -1

Opt, Theta = cheby.problem()

PHI = []#check if this works
for f in cheby.func2d:
    Phi = f(gridk_cheb,Zgrid_cheb)
    PHI.append(Phi)
PHI = np.array(PHI).T
Vg = PHI@Theta
Vl = Vg[:n]
Vh = Vg[n:]
plt.plot(gridk[:n], Vl, 'b')
plt.plot(gridk[:n], Vh, 'r')