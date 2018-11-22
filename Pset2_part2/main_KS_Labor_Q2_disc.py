import numpy as np
import os
os.chdir('C:/Users/DELL/Desktop/Luis/Pset2')
import matplotlib.pyplot as plt
import test3 as ks

ks_g = ks.K_S(Gam = 2, state='bad')
Structure, Lgrid = ks_g.r_w_update()
N = Structure['Intensive Margin Policy']
K = Structure['Optimal Choice Capital']
plt.plot(K)