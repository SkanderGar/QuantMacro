import matplotlib.pyplot as plt
import numpy as np
import os
os.chdir('C:/Users/DELL/Desktop/Quant_macro')
import useful_functions as uf
######## parameters part 1 ##################

order = 4
num_nodes = 20
size_grid = 300
func = uf.chebychev_pol(order)
expo_f = lambda x: np.exp(1/x)
runge_f = lambda x: 1/(1+25*x**2)
ramp_f = lambda x: (x+np.abs(x))/2

############################################
grid = uf.cheb_node(-1,1,size_grid)
nodes = uf.select_node(num_nodes, grid)
para = uf.find_theta_cheb(expo_f,nodes, order)
f_til = lambda x: sum(para[idx]* function(x) for idx, function in enumerate(func))
expo = expo_f(grid)
poly = f_til(grid)
diff = np.abs(np.array(expo)-np.array(poly))
f, (ax,ax_diff) = plt.subplots(1,2)
f.set_figheight(5)
f.set_figwidth(10)
ax.plot(grid, expo, color='r', label='expo')
ax.plot(grid, poly, color='b', label='poly_cub')
ax_diff.plot(grid, diff, color='b', label='diff_cub')
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.set_title('Chebyshev polynomial exp(1/x)')
ax_diff.set_xlabel('x')
ax_diff.set_ylabel('f(x)')
ax_diff.set_title('Chebyshev polynomial error')
ax.set_ylim(-1*10**(5),5*10**(5))
ax_diff.set_ylim(-1*10**(5),5*10**(5))


################################################
grid1 = uf.cheb_node(-1,1,size_grid)
nodes1 = uf.select_node(num_nodes, grid1)
para1 = uf.find_theta_cheb(runge_f,nodes1,order)
f_til1 = lambda x: sum(para1[idx]* function(x) for idx, function in enumerate(func))
runge = runge_f(grid1)
poly1 = f_til1(grid1)
diff1 = np.abs(np.array(runge)-np.array(poly1))
f1, (ax1, ax1_diff) = plt.subplots(1,2)
f1.set_figheight(5)
f1.set_figwidth(10)
ax1.plot(grid1, runge, color='r', label='runge')
ax1.plot(grid1, poly1, color='b', label='poly_cub')
ax1_diff.plot(grid1, diff1, color='b', label='diff_cub')
ax1.set_xlabel('x')
ax1.set_ylabel('f(x)')
ax1.set_title('Chebyshev polynomial 1/(1+25*x**2)')
ax1_diff.set_xlabel('x')
ax1_diff.set_ylabel('f(x)')
ax1_diff.set_title('Chebyshev polynomial error')

#################################################################
grid2 = uf.cheb_node(-1,1,size_grid)
nodes2 = uf.select_node(num_nodes, grid2)
para2 = uf.find_theta_cheb(ramp_f,nodes2,order)
f_til2 = lambda x: sum(para2[idx]* function(x) for idx, function in enumerate(func))
ramp = ramp_f(grid2)
poly2 = f_til2(grid2)
diff2 = np.abs(np.array(ramp)-np.array(poly2))
f2, (ax2, ax2_diff) = plt.subplots(1,2)
f2.set_figheight(5)
f2.set_figwidth(10)
ax2.plot(grid2, ramp, color='r', label='ramp')
ax2.plot(grid2, poly2, color='b', label='poly_cub')
ax2_diff.plot(grid2, diff2, color='b', label='diff_cub')
ax2.set_xlabel('x')
ax2.set_ylabel('f(x)')
ax2.set_title('Chebyshev polynomial (x+|x|)/2')
ax2_diff.set_xlabel('x')
ax2_diff.set_ylabel('f(x)')
ax2_diff.set_title('Chebyshev polynomial error')


#########################################################
###################   monomials  5 ######################
#########################################################

######## parameters order 5 ##################

order_o5 = 6
func_o5 = uf.chebychev_pol(order_o5)
############################################


############################################
para_o5 = uf.find_theta_cheb(expo_f,nodes,order_o5)
f_til_o5 = lambda x: sum(para_o5[idx]* function(x) for idx, function in enumerate(func_o5))
poly_o5 = f_til_o5(grid)
diff_o5 = np.abs(np.array(expo)-np.array(poly_o5))
ax.plot(grid, poly_o5, color='g', label='poly_o5')
ax_diff.plot(grid, diff_o5, color='g', label='diff_o5')
################################################
para1_o5 = uf.find_theta_cheb(runge_f,nodes1,order_o5)
f_til1_o5 = lambda x: sum(para1_o5[idx]* function(x) for idx, function in enumerate(func_o5))
poly1_o5 = f_til1_o5(grid1)
diff1_o5 = np.abs(np.array(runge)-np.array(poly1_o5))
ax1.plot(grid1, poly1_o5, color='g', label='poly_o5')
ax1_diff.plot(grid1, diff1_o5, color='g', label='diff_o5')
#################################################
para2_o5 = uf.find_theta_cheb(ramp_f,nodes2,order_o5)
f_til2_o5 = lambda x: sum(para2_o5[idx]* function(x) for idx, function in enumerate(func_o5))
poly2_o5 = f_til2_o5(grid2)
diff2_o5 = np.abs(np.array(ramp)-np.array(poly2_o5))
ax2.plot(grid2, poly2_o5, color='g', label='poly_o5')
ax2_diff.plot(grid2, diff2_o5, color='g', label='diff_o5')



#########################################################
###################   monomials  10 ######################
#########################################################

######## parameters order 5 ################## diff = np.abs(np.array()-np.array())

order_o10 = 11
func_o10 = uf.chebychev_pol(order_o10)
num_nodes_o10 = 100
############################################

############################################
nodes_o10 = uf.select_node(num_nodes_o10, grid)
para_o10 = uf.find_theta_cheb(expo_f,nodes_o10,order_o10)
f_til_o10 = lambda x: sum(para_o10[idx]* function(x) for idx, function in enumerate(func_o10))
poly_o10 = f_til_o10(grid)
diff_o10 = np.abs(np.array(expo)-np.array(poly_o10))
#ax.plot(grid, poly_o10, color='gold', label='poly_o10')
#ax_diff.plot(grid, diff_o10, color='gold', label='diff_o10')
################################################
nodes1_o10 = uf.select_node(num_nodes_o10, grid1)
para1_o10 = uf.find_theta_cheb(runge_f,nodes1_o10,order_o10)
f_til1_o10 = lambda x: sum(para1_o10[idx]* function(x) for idx, function in enumerate(func_o10))
poly1_o10 = f_til1_o10(grid1)
diff1_o10 = np.abs(np.array(runge)-np.array(poly1_o10))
ax1.plot(grid1, poly1_o10, color='gold', label='poly_o10')
ax1_diff.plot(grid1, diff1_o10, color='gold', label='diff_o10')
#################################################
nodes2_o10 = uf.select_node(num_nodes_o10, grid2)
para2_o10 = uf.find_theta_cheb(ramp_f,nodes2_o10,order_o10)
f_til2_o10 = lambda x: sum(para2_o10[idx]* function(x) for idx, function in enumerate(func_o10))
poly2_o10 = f_til2_o10(grid2)
diff2_o10 = np.abs(np.array(ramp)-np.array(poly2_o10))
ax2.plot(grid2, poly2_o10, color='gold', label='poly_o10')
ax2_diff.plot(grid2, diff2_o10, color='gold', label='diff_o10')



ax_diff.legend(loc = 'upper right')
ax_diff.set_xlabel('x')
ax_diff.set_ylabel('f(x)')

ax1_diff.legend(loc = 'upper right')
ax1_diff.set_xlabel('x')
ax1_diff.set_ylabel('f(x)')

ax2_diff.legend(loc = 'upper right')
ax2_diff.set_xlabel('x')
ax2_diff.set_ylabel('f(x)')


ax.legend(loc='upper right')
ax1.legend(loc='upper right')
ax2.legend(loc='upper right')

