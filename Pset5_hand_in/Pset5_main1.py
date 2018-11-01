import numpy as np
from numpy.random import uniform
import matplotlib.pyplot as plt
directory = "C:/Users/DELL/Desktop/Quant_macro/Pset5"
import os
os.chdir(directory)
import Agent1_bis1 as a

N_a = 100
T = 100
agent = a.agent1(N_a,T,B=1,U2=1)
C_mat, a_Mat = agent.problem()
#### simulation ##
# this simulation is done only with 2 shocks it needs to be adapted for N shocks
start = int(1)
Tr = agent.Tr
#n, c = Tr.shape
shocks = []
shocks.append(start)
rand = uniform(0,1,len(C_mat))

for i in range(len(C_mat)):
    prob = Tr[start,:]
    if rand[i] <= prob[0]:
        next_start = 0
    else:
        next_start = 1
    shocks.append(next_start)
    start = next_start
    
shocks = np.array(shocks)
shocks = np.flip(shocks)#from T to 0 like in the class because of BI


C = []
A = []
for i in range(len(C_mat)):
    c = C_mat[i][shocks[i],0]
    a = a_Mat[i][shocks[i],0]
    C.append(c)
    A.append(a)
C = np.flip(C)
A = np.flip(A)

f, ax = plt.subplots(1,1)
f.set_figheight(5)
f.set_figwidth(10)

ax.plot(C, color = 'b',label = 'Comsumption')
ax.plot(A, color = 'r',label = 'Assets')
ax.legend(loc = 'upper right')
ax.set_xlabel('Period')
ax.set_ylabel('Value')
ax.set_title('Simulation')