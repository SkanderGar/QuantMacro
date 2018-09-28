import matplotlib.pyplot as plt
import numpy as np
import os
os.chdir('C:/Users/DELL/Desktop/Quant_macro')
import useful_functions as uf
######## parameters part 1 ##################
alpha = 1
rho2 = 1/100
rho1 = [1/0.2, 1/0.25]
Px = [lambda x, n=i: np.exp(-alpha*x)/(rho1[n] + rho2*np.exp(-alpha*x)) for i in range(len(rho1))]
order = 4
num_nodes = 6
size_grid = 300
func = uf.chebychev_pol(order)
grid = uf.cheb_node(0,10,size_grid)
nodes = uf.select_node(num_nodes, grid)

############################################

para = uf.find_theta_cheb(Px[0],nodes, order)
f_til = lambda x: sum(para[idx]* function(x) for idx, function in enumerate(func))
Px_0 = Px[0](grid)
poly = f_til(grid)
diff = np.abs(np.array(Px_0)-np.array(poly))
f, (ax,ax_diff) = plt.subplots(1,2)
f.set_figheight(5)
f.set_figwidth(10)
ax.plot(grid, Px_0, color='r', label='rho1=5')
ax.plot(grid, poly, color='b', label='poly_cub')
ax_diff.plot(grid, diff, color='b', label='diff_cub')
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.set_title('Chebyshev polynomial rho1 = 5')
ax_diff.set_xlabel('x')
ax_diff.set_ylabel('f(x)')
ax_diff.set_title('Chebyshev polynomial error')


################################################
para1 = uf.find_theta_cheb(Px[1],nodes,order)
f_til1 = lambda x: sum(para1[idx]* function(x) for idx, function in enumerate(func))
Px_1 = Px[1](grid)
poly1 = f_til1(grid)
diff1 = np.abs(np.array(Px_1)-np.array(poly1))
f1, (ax1, ax1_diff) = plt.subplots(1,2)
f1.set_figheight(5)
f1.set_figwidth(10)
ax1.plot(grid, Px_1, color='r', label='rho1=4')
ax1.plot(grid, poly1, color='b', label='poly_cub')
ax1_diff.plot(grid, diff1, color='b', label='diff_cub')
ax1.set_xlabel('x')
ax1.set_ylabel('f(x)')
ax1.set_title('Chebyshev polynomial rho1 = 4')
ax1_diff.set_xlabel('x')
ax1_diff.set_ylabel('f(x)')
ax1_diff.set_title('Chebyshev polynomial error')


g1, alx1 = plt.subplots(1,1)
g1.set_figheight(5)
g1.set_figwidth(10)
alx1.plot(grid, poly, color='r', label='rho1=5')
alx1.plot(grid, poly1, color='b', label='rho1=4')
alx1.set_xlabel('x')
alx1.set_ylabel('f(x)')
alx1.set_title('Chebyshev order 3 polynomial: 2 cases')







#########################################################
###################   monomials  5 ######################
#########################################################

######## parameters order 5 ##################

order_o5 = 6
func_o5 = uf.chebychev_pol(order_o5)
############################################


############################################
para_o5 = uf.find_theta_cheb(Px[0],nodes,order_o5)
f_til_o5 = lambda x: sum(para_o5[idx]* function(x) for idx, function in enumerate(func_o5))
poly_o5 = f_til_o5(grid)
diff_o5 = np.abs(np.array(Px_0)-np.array(poly_o5))
ax.plot(grid, poly_o5, color='g', label='poly_o5')
ax_diff.plot(grid, diff_o5, color='g', label='diff_o5')
################################################
para1_o5 = uf.find_theta_cheb(Px[1],nodes,order_o5)
f_til1_o5 = lambda x: sum(para1_o5[idx]* function(x) for idx, function in enumerate(func_o5))
poly1_o5 = f_til1_o5(grid)
diff1_o5 = np.abs(np.array(Px_1)-np.array(poly1_o5))
ax1.plot(grid, poly1_o5, color='g', label='poly_o5')
ax1_diff.plot(grid, diff1_o5, color='g', label='diff_o5')



g2, alx2 = plt.subplots(1,1)
g2.set_figheight(5)
g2.set_figwidth(10)
alx2.plot(grid, poly_o5, color='r', label='rho1=5')
alx2.plot(grid, poly1_o5, color='b', label='rho1=4')
alx2.set_xlabel('x')
alx2.set_ylabel('f(x)')
alx2.set_title('Chebyshev order 5 polynomial: 2 cases')
#########################################################
###################   monomials  10 ######################
#########################################################

######## parameters order 5 ################## diff = np.abs(np.array()-np.array())

order_o10 = 11
func_o10 = uf.chebychev_pol(order_o10)
num_nodes_o10 = 20
############################################

############################################
nodes_o10 = uf.select_node(num_nodes_o10, grid)
para_o10 = uf.find_theta_cheb(Px[0],nodes_o10,order_o10)
f_til_o10 = lambda x: sum(para_o10[idx]* function(x) for idx, function in enumerate(func_o10))
poly_o10 = f_til_o10(grid)
diff_o10 = np.abs(np.array(Px_0)-np.array(poly_o10))
ax.plot(grid, poly_o10, color='gold', label='poly_o10')
ax_diff.plot(grid, diff_o10, color='gold', label='diff_o10')
################################################
nodes1_o10 = uf.select_node(num_nodes_o10, grid)
para1_o10 = uf.find_theta_cheb(Px[1],nodes1_o10,order_o10)
f_til1_o10 = lambda x: sum(para1_o10[idx]* function(x) for idx, function in enumerate(func_o10))
poly1_o10 = f_til1_o10(grid)
diff1_o10 = np.abs(np.array(Px_1)-np.array(poly1_o10))
ax1.plot(grid, poly1_o10, color='gold', label='poly_o10')
ax1_diff.plot(grid, diff1_o10, color='gold', label='diff_o10')

##########################
g3, alx3 = plt.subplots(1,1)
g3.set_figheight(5)
g3.set_figwidth(10)
alx3.plot(grid, poly_o10, color='r', label='rho1=5')
alx3.plot(grid, poly1_o10, color='b', label='rho1=4')
alx3.set_xlabel('x')
alx3.set_ylabel('f(x)')
alx3.set_title('Chebyshev order 10 polynomial: 2 cases')


alx1.legend(loc = 'upper right')
alx2.legend(loc = 'upper right')
alx3.legend(loc = 'upper right')


ax_diff.legend(loc = 'upper right')
ax_diff.set_xlabel('x')
ax_diff.set_ylabel('f(x)')

ax1_diff.legend(loc = 'upper right')
ax1_diff.set_xlabel('x')
ax1_diff.set_ylabel('f(x)')

ax.legend(loc='upper right')
ax1.legend(loc='upper right')




