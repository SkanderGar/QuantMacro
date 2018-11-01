import numpy as np
from numpy.random import uniform
import matplotlib.pyplot as plt
directory = "C:/Users/DELL/Desktop/Quant_macro/Pset5"
import os
os.chdir(directory)
import Agent1_23 as a

N_a = 100
## T is defined as a default parameter
agent = a.agent1(N_a)
V, Ga, Gc = agent.problem()
####partial equilibrium 4.1.2####
###certainty part
agent2 = a.agent1(N_a, Sig = 2, C_ = 100, cert = 1)
V_cer, Ga_cer, Gc_cer = agent2.problem()

f, (ax, ax1) = plt.subplots(1,2)
f.set_figheight(5)
f.set_figwidth(10)

ax.plot(agent.grid_a,Gc[40][:,0], color = 'b',label = 'Comsumption age 5 Low')#because of BI
ax.plot(agent.grid_a,Gc[40][:,1], color = 'r',label = 'Comsumption age 5 high')
ax.plot(agent.grid_a,Gc[5][:,0], color = 'g',label = 'Comsumption age 40 Low')
ax.plot(agent.grid_a,Gc[5][:,1], color = 'k',label = 'Comsumption age 40 high')
ax.legend(loc = 'upper right')
ax.set_xlabel('Period')
ax.set_ylabel('Value')
ax.set_title('Uncertainty')

ax1.plot(agent.grid_a,Gc_cer[40], color = 'b',label = 'Comsumption age 5')#because of BI
ax1.plot(agent.grid_a,Gc_cer[5], color = 'r',label = 'Comsumption age 40')
ax1.legend(loc = 'upper right')
ax1.set_xlabel('Period')
ax1.set_ylabel('Value')
ax1.set_title('Certainty')
