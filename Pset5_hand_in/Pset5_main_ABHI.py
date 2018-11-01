import numpy as np
from numpy.random import uniform
import matplotlib.pyplot as plt
directory = "C:/Users/DELL/Desktop/Quant_macro/Pset5"
import os
os.chdir(directory)
import ABHI as a

N_a = 200
## T is defined as a default parameter
agent = a.agent1(N_a, Mu_y = 3, gamma_y = 0.6, Sig = 5)
Structure = agent.Interest_update(num_r = 60, r_min = 0.03, r_max = 0.08, pas = 0.5)
grida = agent.grid_a

f, (ax, ax1) = plt.subplots(1,2)
f.set_figheight(5)
f.set_figwidth(12)

ax.plot(grida, Structure['smoothed_dist_h'], color = 'b')
ax.set_xlabel('Assets Today')
ax.set_ylabel('Density')
ax.set_title('Distribution High State')

ax1.plot(grida, Structure['smoothed_dist_l'], color = 'b')
ax1.set_xlabel('Assets Today')
ax1.set_ylabel('Density')
ax1.set_title('Distribution Low State')


### preparing distribution of consumption
g_c = Structure['gc']
f2, (ax3, ax4) = plt.subplots(1,2)
f2.set_figheight(5)
f2.set_figwidth(12)

ax3.plot(g_c[:,0], Structure['smoothed_dist_h'], color = 'b')
ax3.set_xlabel('Consumption')
ax3.set_ylabel('Density')
ax3.set_title('Distribution High State')

ax4.plot(g_c[:,1], Structure['smoothed_dist_l'], color = 'b')
ax4.set_xlabel('Consumption')
ax4.set_ylabel('Density')
ax4.set_title('Distribution Low State')