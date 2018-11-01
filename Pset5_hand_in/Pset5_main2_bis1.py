import numpy as np
from numpy.random import uniform
import matplotlib.pyplot as plt
directory = "C:/Users/DELL/Desktop/Quant_macro/Pset5"
import os
os.chdir(directory)
import Aiyag_cont_1 as a

N_a = 30
agent = a.rep_agent(N_a, B=0)
V_interp, g_c, g_a = agent.problem()
grida = agent.grid_a
vl = V_interp[0](grida)
vh = V_interp[1](grida)
V_new = np.vstack((vl,vh)).T 

f, (ax, ax1) = plt.subplots(1,2)
f.set_figheight(5)
f.set_figwidth(12)

ax.plot(grida, V_new[:,0], color = 'b',label = 'Value Low')
ax.plot(grida, V_new[:,1], color = 'r',label = 'Value High')
ax.legend(loc = 'upper right')
ax.set_xlabel('Assets Today')
ax.set_ylabel('Value')
ax.set_title('Variables')

ax1.plot(grida, g_c[:,0], color = 'b',label = 'Consumption Low')
ax1.plot(grida, g_c[:,1], color = 'r',label = 'Consumption High')
ax1.legend(loc = 'upper right')
ax1.set_xlabel('Assets Today')
ax1.set_ylabel('Value')
ax1.set_title('Variables')


A, C = agent.simulation(g_c, g_a)

f2, ax2 = plt.subplots(1,1)
f2.set_figheight(5)
f2.set_figwidth(10)

ax2.plot(C, color = 'b',label = 'Consumption')
ax2.plot(A, color = 'r',label = 'Assets')
ax2.legend(loc = 'upper right')
ax2.set_xlabel('Time')
ax2.set_ylabel('Value')
ax2.set_title('Simulation')



