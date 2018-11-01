import numpy as np
from numpy.random import uniform
import matplotlib.pyplot as plt
directory = "C:/Users/DELL/Desktop/Quant_macro/Pset5"
import os
os.chdir(directory)
import Aiyag_disc_1 as a

N_a = 100
agent = a.agent1(N_a, sigma_y = 0.5, gamma = 0.95, Sig = 5)
grida = agent.grid_a
V_new, ga, gc = agent.problem()

f, (ax, ax1, ax2) = plt.subplots(1,3)
f.set_figheight(5)
f.set_figwidth(12)

ax.plot(grida, V_new[:,0], color = 'b',label = 'Value Low')
ax.plot(grida, V_new[:,1], color = 'r',label = 'Value High')
ax.legend(loc = 'upper right')
ax.set_xlabel('Assets Today')
ax.set_ylabel('Value')
ax.set_title('Variables')

ax1.plot(grida, gc[:,0], color = 'b',label = 'Comsumption Low')
ax1.plot(grida, gc[:,1], color = 'r',label = 'Comsumption High')
ax1.legend(loc = 'upper right')
ax1.set_xlabel('Assets Today')
ax1.set_ylabel('Value')
ax1.set_title('Variables')

ax2.plot(grida, ga[:,0], color = 'b',label = 'Assets Low')
ax2.plot(grida, ga[:,1], color = 'r',label = 'Assets High')
ax2.legend(loc = 'upper right')
ax2.set_xlabel('Assets Today')
ax2.set_ylabel('Value')
ax2.set_title('Variables')


A, C = agent.simulation(gc, ga)
f2, ax3 = plt.subplots(1,1)
f2.set_figheight(5)
f2.set_figwidth(10)

ax3.plot(A, color = 'b',label = 'Assets')
ax3.plot(C, color = 'r',label = 'Consumption')
ax3.legend(loc = 'upper right')
ax3.set_xlabel('Time')
ax3.set_ylabel('Value')
ax3.set_title('Simulation')


