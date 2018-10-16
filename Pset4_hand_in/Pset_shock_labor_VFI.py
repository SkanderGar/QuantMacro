import numpy as np
import os
os.chdir('C:/Users/DELL/Desktop/Quant_macro/Pset4')
import rep_VFI_shock_labor as ra
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
n = 50
kmax = kss
kmin = 0.1*kss
hmin = 0
hmax = 1

###steady state

rep_age = ra.rep_agent(para['theta'], para['beta'], para['delta'], para['kappa'], para['nu'], kmin, kmax, hmin, hmax, n=n)
V, gk, gc, gl = rep_age.problem()

f, (ax1, ax) = plt.subplots(1,2)
f.set_figheight(5)
f.set_figwidth(10)

ax1.plot(rep_age.gridk,V[:,0], 'r', label='Bad Shock')
ax1.plot(rep_age.gridk,V[:,1], 'b', label='Good Shock')
ax1.legend(loc = 'upper right')
ax1.set_xlabel('k')
ax1.set_ylabel('V')
ax1.set_title('Value Function')

K, C, Y, I, L, W, LS = rep_age.simulation(T=200)

ax.plot(K, 'b', label='Capital')
ax.legend(loc = 'upper right')
ax.set_xlabel('Time')
ax.set_ylabel('Level')
ax.set_title('Variables')

fb, (ax1b, ax2b, ax3b) = plt.subplots(1,3)
fb.set_figheight(5)
fb.set_figwidth(10)

ax1b.plot(I, 'b', label='Investment')
ax1b.legend(loc = 'upper right')
ax1b.set_xlabel('Time')
ax1b.set_ylabel('Level')
ax1b.set_title('Variables')

ax2b.plot(Y, 'b', label='Output')
ax2b.legend(loc = 'upper right')
ax2b.set_xlabel('Time')
ax2b.set_ylabel('Level')
ax2b.set_title('Variables')

ax3b.plot(C, 'b', label='Consumption')
ax3b.legend(loc = 'upper right')
ax3b.set_xlabel('Time')
ax3b.set_ylabel('Level')
ax3b.set_title('Variables')


f2, (ax3, ax4, ax5) = plt.subplots(1,3)
f2.set_figheight(5)
f2.set_figwidth(10)

ax3.plot(L, 'b', label='Labor supply')
ax3.legend(loc = 'upper right')
ax3.set_xlabel('Time')
ax3.set_ylabel('Level')
ax3.set_title('Labor Supply')

ax4.plot(W, 'b', label='Wages')
ax4.legend(loc = 'upper right')
ax4.set_xlabel('Time')
ax4.set_ylabel('Level')
ax4.set_title('Wages')

ax5.plot(LS, 'b', label='Labor Share')
ax5.legend(loc = 'upper right')
ax5.set_xlabel('Time')
ax5.set_ylabel('Level')
ax5.set_title('Labor Share')


K_ir, L_ir, C_ir, Y_ir, I_ir, W_ir = rep_age.Impulse_resp(kss)
g, (gx1, gx2, gx3) = plt.subplots(1,3)
g.set_figheight(5)
g.set_figwidth(10)

gx1.plot(K_ir, 'b', label='Impulse Capital')
gx1.legend(loc = 'upper right')
gx1.set_xlabel('Time')
gx1.set_ylabel('Level')
gx1.set_title('Impulse Capital')

gx2.plot(L_ir, 'b', label='Impulse Labor')
gx2.legend(loc = 'upper right')
gx2.set_xlabel('Time')
gx2.set_ylabel('Level')
gx2.set_title('Impulse Labor')

gx3.plot(C_ir, 'b', label='Impulse Consumption')
gx3.legend(loc = 'upper right')
gx3.set_xlabel('Time')
gx3.set_ylabel('Level')
gx3.set_title('Impulse Consumption')

g2, (gx4, gx5, gx6) = plt.subplots(1,3)
g2.set_figheight(5)
g2.set_figwidth(10)

gx4.plot(Y_ir, 'b', label='Impulse Output')
gx4.legend(loc = 'upper right')
gx4.set_xlabel('Time')
gx4.set_ylabel('Level')
gx4.set_title('Impulse Output')

gx5.plot(I_ir, 'b', label='Impulse Investment')
gx5.legend(loc = 'upper right')
gx5.set_xlabel('Time')
gx5.set_ylabel('Level')
gx5.set_title('Impulse Investment')

gx6.plot(W_ir, 'b', label='Impulse Wage')
gx6.legend(loc = 'upper right')
gx6.set_xlabel('Time')
gx6.set_ylabel('Level')
gx6.set_title('Impulse Wage')







