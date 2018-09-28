import numpy as np
import scipy.interpolate as interp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import contour

import matplotlib.pyplot as plt
import numpy as np
import os
os.chdir('C:/Users/DELL/Desktop/Quant_macro')
import useful_functions as uf
######## parameters part 1 ##################
order = 4
start = 0
num_grid_x = 10
num_node_x = 5
num_grid_y = 8
num_node_y = 4
sig_el = 2
Total_num_nodes = num_node_x + num_node_y
Total_num_grid = num_grid_x + num_grid_y
grid_x = uf.cheb_node(0,10,num_grid_x)
nodes_x = uf.select_node(num_node_x, grid_x)
grid_y = uf.cheb_node(0,10,num_grid_y)
nodes_y = uf.select_node(num_node_y, grid_y)

alpha = 0.5
sigma = [0.25, 5, 10] # I think there is a typo because not defined at 1
f = [lambda k, h, n=i: ((1-alpha)*k**((sigma[n]-1)/sigma[n]) +
                        alpha*h**((sigma[n]-1)/sigma[n]))**(sigma[n]/(sigma[n]-1)) for i in range(len(sigma))]

func = uf.chebychev_pol_2dim(order, start)

Node = []
Grid = []

for i in range(len(nodes_x)):
    for j in range(len(nodes_y)):
        point = [nodes_x[i], nodes_y[j]]
        Node.append(point)

for i in range(len(grid_x)):
    for j in range(len(grid_y)):
        point = [grid_x[i], grid_y[j]]
        Grid.append(point)
        
param = uf.find_theta_cheb_2dim(f[sig_el],Node, order, start)
f_til = lambda x, y: sum(param[idx]* function(x,y) for idx, function in enumerate(func))#no issue because defined inside the function

Y_0 = []
Y_til_0 = []
for element in Grid:
    Y_0.append(f[sig_el](element[0], element[1]))
    Y_til_0.append(f_til(element[0], element[1]))
    

X = np.array(Grid)[:,0]
Y = np.array(Grid)[:,1]
Z = np.array(Y_til_0)
Z_diff = np.abs(np.array(Y_0)-np.array(Y_til_0))

plotx,ploty, = np.meshgrid(np.linspace(np.min(X),np.max(X),10),\
                           np.linspace(np.min(Y),np.max(Y),10))
plotz = interp.griddata((X,Y),Z_diff,(plotx,ploty),method='linear')

fig = plt.figure()
fig.set_figheight(5)
fig.set_figwidth(10)#it was 13
ax = fig.add_subplot(131, projection='3d')
ax.plot_surface(plotx,ploty,plotz,cstride=1,rstride=1,cmap='viridis')
###############################subplot2
ax1 = fig.add_subplot(132, projection='3d')
plotz1 = interp.griddata((X,Y),Y_0,(plotx,ploty),method='linear')
ax1.plot_surface(plotx,ploty,plotz1,cstride=1,rstride=1,cmap='viridis')

#################################subplot3
ax2 = fig.add_subplot(133, projection='3d')
plotz2 = interp.griddata((X,Y),Y_til_0,(plotx,ploty),method='linear')
ax2.plot_surface(plotx,ploty,plotz2,cstride=1,rstride=1,cmap='viridis')



ax.set_xlabel('k')
ax.set_ylabel('h')
ax.set_zlabel('f(k,h)')
ax.set_title('Error in Chebychev interpolation')


ax1.set_xlabel('k')
ax1.set_ylabel('h')
ax1.set_zlabel('f(k,h)')
ax1.set_title('Real function')


ax2.set_xlabel('k')
ax2.set_ylabel('h')
ax2.set_zlabel('f(k,h)')
ax2.set_title('Chebychev interpolation order 15')

####################isoquants#########################

F_iso = [np.max(Z)*0.05, np.max(Z)*0.1, np.max(Z)*0.25, np.max(Z)*0.5, np.max(Z)*0.75, np.max(Z)*0.9, np.max(Z)*0.95]
X1 = np.linspace(np.min(X),np.max(X),10)
Y1 = np.linspace(np.min(X),np.max(X),10)
g = plt.figure()
g.set_figheight(5)
g.set_figwidth(10)
cont_ax = g.add_subplot(121)
cont = contour(X1,Y1,plotz2,F_iso)
plt.clabel(cont,inline = 1, fontsize=10)
cont_ax.set_xlabel('Capital')
cont_ax.set_ylabel('Labor')
cont_ax.set_title('Isoquants approximation sigma=10')
cont_ax1 = g.add_subplot(122)
cont = contour(X1,Y1,plotz1,F_iso)
plt.clabel(cont,inline = 1, fontsize=10)
cont_ax1.set_xlabel('Capital')
cont_ax1.set_ylabel('Labor')
cont_ax1.set_title('Real isoquants sigma=10')


####################computing the error in isoquants###################
Iso_ks, Iso_hs = [], []
for F_bar in F_iso:
    Iso_k, Iso_h = uf.Iso_quants(f[sig_el], F_bar, 100)
    Iso_ks.append(Iso_k)
    Iso_hs.append(Iso_h)

quantv_til = []
qu_in = []    
for quantile in range(len(Iso_ks)):
    for idx in range(len(Iso_ks[quantile])):
        level_f = f[sig_el](Iso_ks[quantile][idx],Iso_hs[quantile][idx])
        qu_in.append(level_f)
    quantv_til.append(qu_in)
    qu_in = []
    
errors_quant = []
errors = []

for quant in range(len(quantv_til)):
    for idx, est_q in enumerate(quantv_til[quant]):
        error = F_iso[quant]-est_q
        errors.append(error)
    errors_quant.append(errors)
    errors = []
hig0 = plt.figure()
hig0.set_figheight(5)
hig0.set_figwidth(10)
hax0 = hig0.add_subplot(331)
hax1 = hig0.add_subplot(332)
hax2 = hig0.add_subplot(333)

hig1 = plt.figure()
hig1.set_figheight(5)
hig1.set_figwidth(10)
hax3 = hig1.add_subplot(331)
hax4 = hig1.add_subplot(332)
hax5 = hig1.add_subplot(333)

hig2 = plt.figure()
hig2.set_figheight(5)
hig2.set_figwidth(10)
hax6 = hig2.add_subplot(331)



colors = ['b', 'r', 'g', 'k', 'c', 'gold', 'tomato']
hax0.scatter(np.arange(0,len(errors_quant[0]),1),errors_quant[0], marker=".", color = colors[0],s=4)
hax1.scatter(np.arange(0,len(errors_quant[1]),1),errors_quant[1], marker=".", color = colors[1],s=4)
hax2.scatter(np.arange(0,len(errors_quant[2]),1),errors_quant[2], marker=".", color = colors[2],s=4)
hax3.scatter(np.arange(0,len(errors_quant[3]),1),errors_quant[3], marker=".", color = colors[3],s=4)
hax4.scatter(np.arange(0,len(errors_quant[4]),1),errors_quant[4], marker=".", color = colors[4],s=4)
hax5.scatter(np.arange(0,len(errors_quant[5]),1),errors_quant[5], marker=".", color = colors[5],s=4)
hax6.scatter(np.arange(0,len(errors_quant[6]),1),errors_quant[6], marker=".", color = colors[6],s=4)

hax0.set_xlabel('Elements of the list')
hax0.set_ylabel('Error')
hax0.set_title('Error in percentile 0.05')

hax1.set_xlabel('Elements of the list')
hax1.set_ylabel('Error')
hax1.set_title('Error in percentile 0.1')

hax2.set_xlabel('Elements of the list')
hax2.set_ylabel('Error')
hax2.set_title('Error in percentile 0.25')

hax3.set_xlabel('Elements of the list')
hax3.set_ylabel('Error')
hax3.set_title('Error in percentile 0.5')

hax4.set_xlabel('Elements of the list')
hax4.set_ylabel('Error')
hax4.set_title('Error in percentile 0.75')

hax5.set_xlabel('Elements of the list')
hax5.set_ylabel('Error')
hax5.set_title('Error in percentile 0.9')

hax6.set_xlabel('Elements of the list')
hax6.set_ylabel('Error')
hax6.set_title('Error in percentile 0.95')

#hig0.savefig('Err_quant3s0.png')
#hig1.savefig('Err_quant3s1.png')
#hig2.savefig('Err_quant3s2.png')





        
        
        
        
        
        
        
        