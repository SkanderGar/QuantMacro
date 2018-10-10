#point 1
import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir('C:/Users/DELL/Desktop/last chance')
import Rep_agent_tr as RA
#parameters
n = 200
Kss1 = 4
Kmax = Kss1
Kmin = 0.1*Kss1

par = {}
par['theta'] = 0.67
par['h'] = 0.31
par['beta'] = 0.98
par['delta'] = 0.25/4
par['z1'] = Kss1/(par['h']*((par['beta']*(1-par['theta']))/(1-par['beta']*(1-par['delta'])))**(1/par['theta']))
par['z2'] = par['z1']*2
Kss2 = par['z2']*(par['h']*((par['beta']*(1-par['theta']))/(1-par['beta']*(1-par['delta'])))**(1/par['theta']))

Kmax1 = 2*Kss2
Kmin1 = 0.1*Kss1 #because we don't want to exceed Kss1 for question D

############Question C #############
gridk1 = np.linspace(Kmin1, Kmax1, n)
K = np.tile(gridk1,(n,1)).T
K_p = K.T
Vstart = np.zeros((n,))

model0 = RA.model1(par['z1'])
V0, g_c0, g_k0 = model0.problem(Vstart, K, K_p)
model1 = RA.model1(par['z2'])
Vf, Kf = model1.trans(V0, g_k0, K, K_p)

Ktr = []
Ytr = []
Ctr = []
itr = []
Ktr.append(Kss1)
Ytr.append(Kss1**(1-par['theta']) *(par['z2']*par['h'])**par['theta'])
itr.append(par['delta']*Kss1)
Cnext = Ytr[-1]-itr[-1]
Ctr.append(Cnext)
for idx in range(1,len(Kf)):
    Ktr.append(Kf[idx](Ktr[idx-1]))
    Ytr.append(Ktr[-1]**(1-par['theta']) *(par['z2']*par['h'])**par['theta'])
    itr.append(Ktr[-1]-(1-par['delta'])*Ktr[-2])
    Cnext = Ytr[-1]-itr[-1]
    Ctr.append(Cnext)

######################question d#######
pol_10 = Kf[:10]

Kd = []
Yd = []
Cd = []
Id = []
Kd.append(Kss1)
Yd.append(Kd[-1]**(1-par['theta']) *(par['z2']*par['h'])**par['theta'])
Id.append(par['delta']*Kd[-1])
Cdnext = Yd[-1]-Id[-1]
Cd.append(Cdnext)
for fu in pol_10:
    Kd.append(fu(Kd[-1]))
    Yd.append(Kd[-1]**(1-par['theta']) *(par['z2']*par['h'])**par['theta'])
    Id.append(Kd[-1]-(1-par['delta'])*Kd[-2])
    Cdnext = Yd[-1]-Id[-1]
    Cd.append(Cdnext)

K10 = Kf[10](gridk1)    
V10 = Vf[10](gridk1)
V_sho, K_sho = model0.trans(V10, K10, K, K_p)
for idx in range(1,len(K_sho)):
    Kd.append(K_sho[idx](Kd[-1]))
    Yd.append(Kd[-1]**(1-par['theta']) *(par['z2']*par['h'])**par['theta'])
    Id.append(Kd[-1]-(1-par['delta'])*Kd[-2])
    Cdnext = Yd[-1]-Id[-1]
    Cd.append(Cdnext)
    
    
###################plots##################

f, (ax1,ax2) = plt.subplots(1,2)
f.set_figheight(5)
f.set_figwidth(10)

ax1.plot(Ctr,color='b', label = 'C')
ax1.plot(Ktr,color='r', label = 'K')
ax1.plot(itr,color='g', label = 'i')
ax1.plot(Ytr,color='k', label = 'Y')
ax1.legend(loc = 'upper right')
ax1.set_ylabel('Level')
ax1.set_xlabel('Time')
ax1.set_title('Transition from SS1 to SS2')

ax2.plot(Cd,color='b', label = 'C')
ax2.plot(Kd,color='r', label = 'K')
ax2.plot(Id,color='g', label = 'i')
ax2.plot(Yd,color='k', label = 'Y')
ax2.legend(loc = 'upper right')
ax2.set_ylabel('Level')
ax2.set_xlabel('Time')
ax2.set_title('Transition from SS1 to SS2 then SS2 to SS1')



     








